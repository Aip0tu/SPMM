import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, WordpieceTokenizer


from dataset import SMILESDataset_FluoDB
from scheduler import create_scheduler
from xbert import BertConfig, BertForMaskedLM


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class SPMM_fluodb_regressor(nn.Module):
    def __init__(self, tokenizer=None, config=None):
        super().__init__()
        self.tokenizer = tokenizer

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        for i in range(bert_config.fusion_layer, bert_config.num_hidden_layers):
            self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()
        text_width = self.text_encoder.config.hidden_size

        self.reg_head = nn.Sequential(
            nn.Linear(text_width * 4, text_width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_width * 2, 1)
        )

    def encode(self, input_ids, attention_mask):
        return self.text_encoder.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            mode='text',
        ).last_hidden_state[:, 0, :]

    def forward(self, smiles_input_ids, smiles_attention_mask, solvent_input_ids, solvent_attention_mask, value=None, eval=False):
        mol_embeddings = self.encode(smiles_input_ids, smiles_attention_mask)
        solvent_embeddings = self.encode(solvent_input_ids, solvent_attention_mask)
        fused_embeddings = torch.cat(
            [
                mol_embeddings,
                solvent_embeddings,
                torch.abs(mol_embeddings - solvent_embeddings),
                mol_embeddings * solvent_embeddings,
            ],
            dim=-1,
        )
        pred = self.reg_head(fused_embeddings).squeeze(-1)

        if eval:
            return pred
        return nn.MSELoss()(pred, value)


def build_tokenizer(vocab_filename):
    tokenizer = BertTokenizer(vocab_file=vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab,
        unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250,
    )
    return tokenizer


def tokenize_pair(tokenizer, smiles, solvents, device, max_length_smiles, max_length_solvent):
    smiles_input = tokenizer(smiles, padding='longest', truncation=True, max_length=max_length_smiles, return_tensors='pt').to(device)
    solvent_input = tokenizer(solvents, padding='longest', truncation=True, max_length=max_length_solvent, return_tensors='pt').to(device)
    return smiles_input, solvent_input


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, max_length_smiles, max_length_solvent):
    model.train()

    header = f'Train Epoch: [{epoch}]'
    print_freq = 20
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (smiles, solvents, value) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        value = value.to(device, non_blocking=True)
        smiles_input, solvent_input = tokenize_pair(
            tokenizer,
            smiles,
            solvents,
            device,
            max_length_smiles,
            max_length_solvent,
        )

        loss = model(
            smiles_input.input_ids[:, 1:],
            smiles_input.attention_mask[:, 1:],
            solvent_input.input_ids[:, 1:],
            solvent_input.attention_mask[:, 1:],
            value,
        )
        loss.backward()
        optimizer.step()

        tqdm_data_loader.set_description(f"loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, denormalize, max_length_smiles, max_length_solvent):
    model.eval()
    preds = []
    answers = []
    for smiles, solvents, value in data_loader:
        value = value.to(device, non_blocking=True)
        smiles_input, solvent_input = tokenize_pair(
            tokenizer,
            smiles,
            solvents,
            device,
            max_length_smiles,
            max_length_solvent,
        )

        prediction = model(
            smiles_input.input_ids[:, 1:],
            smiles_input.attention_mask[:, 1:],
            solvent_input.input_ids[:, 1:],
            solvent_input.attention_mask[:, 1:],
            value,
            eval=True,
        )

        preds.append(prediction.cpu())
        answers.append(value.cpu())

    preds = torch.cat(preds, dim=0)
    answers = torch.cat(answers, dim=0)

    value_mean, value_std = denormalize
    preds = preds * value_std + value_mean
    answers = answers * value_std + value_mean

    rmse = torch.sqrt(nn.MSELoss()(preds, answers)).item()
    mae = torch.mean(torch.abs(preds - answers)).item()
    r2 = r2_score(answers.numpy(), preds.numpy())
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def resolve_dataset(task, data_dir):
    train_path = os.path.join(data_dir, f'{task}_train.csv')
    valid_path = os.path.join(data_dir, f'{task}_valid.csv')
    test_path = os.path.join(data_dir, f'{task}_test.csv')
    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return train_path, valid_path, test_path


def main(args, config):
    device = torch.device(args.device)
    print('TASK:', args.task)
    print('DATA DIR:', args.data_dir)

    train_path, valid_path, test_path = resolve_dataset(args.task, args.data_dir)
    dataset_train = SMILESDataset_FluoDB(train_path, target_name=args.task)
    dataset_val = SMILESDataset_FluoDB(
        valid_path,
        target_name=args.task,
        value_mean=dataset_train.value_mean,
        value_std=dataset_train.value_std,
    )
    dataset_test = SMILESDataset_FluoDB(
        test_path,
        target_name=args.task,
        value_mean=dataset_train.value_mean,
        value_std=dataset_train.value_std,
    )

    print(len(dataset_train), len(dataset_val), len(dataset_test))
    train_loader = DataLoader(
        dataset_train,
        batch_size=config['batch_size_train'],
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=config['batch_size_test'],
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=config['batch_size_test'],
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    tokenizer = build_tokenizer(args.vocab_filename)

    seed = args.seed if args.seed else random.randint(0, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print('Creating model')
    model = SPMM_fluodb_regressor(config=config, tokenizer=tokenizer)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    if args.freeze_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        print('Encoder frozen')

    model = model.to(device)
    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best_valid = float('inf')
    best_test = None

    start_time = time.time()
    for epoch in range(max_epoch):
        print('TRAIN', epoch)
        train(
            model,
            train_loader,
            optimizer,
            tokenizer,
            epoch,
            warmup_steps,
            device,
            lr_scheduler,
            args.max_length_smiles,
            args.max_length_solvent,
        )
        denormalize = (dataset_train.value_mean, dataset_train.value_std)
        val_stats = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            denormalize,
            args.max_length_smiles,
            args.max_length_solvent,
        )
        test_stats = evaluate(
            model,
            test_loader,
            tokenizer,
            device,
            denormalize,
            args.max_length_smiles,
            args.max_length_solvent,
        )
        print('VALID RMSE: %.4f | MAE: %.4f | R2: %.4f' % (val_stats['rmse'], val_stats['mae'], val_stats['r2']))
        print('TEST  RMSE: %.4f | MAE: %.4f | R2: %.4f' % (test_stats['rmse'], test_stats['mae'], test_stats['r2']))

        if val_stats['rmse'] < best_valid:
            best_valid = val_stats['rmse']
            best_test = test_stats
            save_obj = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'task': args.task,
                'value_mean': dataset_train.value_mean,
                'value_std': dataset_train.value_std,
            }
            save_path = os.path.join(args.output_dir, f'{args.task}_best.pth')
            torch.save(save_obj, save_path)
            print('SAVED:', save_path)

        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if best_test is not None:
        print(
            'TASK:',
            args.task,
            '\tBest test metrics from the checkpoint with best validation RMSE:',
            best_test,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='emi', choices=['abs', 'emi', 'plqy', 'e'])
    parser.add_argument('--data_dir', default='./FlourDB')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--output_dir', default='./output/FluoDB')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--min_lr', default=3e-6, type=float)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_length_smiles', default=128, type=int)
    parser.add_argument('--max_length_solvent', default=64, type=int)
    parser.add_argument('--freeze_encoder', action='store_true')
    args = parser.parse_args()

    cls_config = {
        'batch_size_train': args.batch_size,
        'batch_size_test': 64,
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'schedular': {
            'sched': 'cosine',
            'lr': args.lr,
            'epochs': args.epoch,
            'min_lr': args.min_lr,
            'decay_rate': 1,
            'warmup_lr': 0.5e-5,
            'warmup_epochs': 1,
            'cooldown_epochs': 0,
        },
        'optimizer': {'opt': 'adamW', 'lr': args.lr, 'weight_decay': 0.02},
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cls_config)

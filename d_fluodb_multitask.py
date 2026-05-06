import argparse
import datetime
import os
import random
import sys
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
from xbert import BertConfig, BertForMaskedLM

from dataset import SMILESDataset_FluoDB_MultiTask
from d_fluodb_regression import build_tokenizer, tokenize_pair
from scheduler import create_scheduler


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')


class SPMM_fluodb_multitask_regressor(nn.Module):
    def __init__(self, tokenizer=None, config=None, solvent_desc_dim=9, n_targets=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_targets = n_targets

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        for i in range(bert_config.fusion_layer, bert_config.num_hidden_layers):
            self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()

        text_width = self.text_encoder.config.hidden_size
        fusion_heads = config.get('fusion_heads', 8)
        fusion_dropout = config.get('fusion_dropout', 0.1)
        solvent_desc_width = config.get('solvent_desc_width', 128)

        self.mol_to_solvent_attn = nn.MultiheadAttention(
            embed_dim=text_width,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            batch_first=True,
        )
        self.solvent_to_mol_attn = nn.MultiheadAttention(
            embed_dim=text_width,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            batch_first=True,
        )
        self.fusion_norm_mol = nn.LayerNorm(text_width)
        self.fusion_norm_solvent = nn.LayerNorm(text_width)

        self.solvent_desc_proj = nn.Sequential(
            nn.Linear(solvent_desc_dim, solvent_desc_width),
            nn.GELU(),
            nn.LayerNorm(solvent_desc_width),
        )

        fused_width = text_width * 6 + solvent_desc_width
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_width, text_width * 2),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(text_width * 2, text_width),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(text_width, 1),
            )
            for _ in range(n_targets)
        ])

    def encode_tokens(self, input_ids, attention_mask):
        return self.text_encoder.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            mode='text',
        ).last_hidden_state

    def forward(self, smiles_input_ids, smiles_attention_mask, solvent_input_ids,
                solvent_attention_mask, solvent_desc, value=None, target_mask=None, eval=False):
        mol_tokens = self.encode_tokens(smiles_input_ids, smiles_attention_mask)
        solvent_tokens = self.encode_tokens(solvent_input_ids, solvent_attention_mask)

        mol_padding_mask = smiles_attention_mask == 0
        solvent_padding_mask = solvent_attention_mask == 0
        mol_context, _ = self.mol_to_solvent_attn(
            query=mol_tokens,
            key=solvent_tokens,
            value=solvent_tokens,
            key_padding_mask=solvent_padding_mask,
            need_weights=False,
        )
        solvent_context, _ = self.solvent_to_mol_attn(
            query=solvent_tokens,
            key=mol_tokens,
            value=mol_tokens,
            key_padding_mask=mol_padding_mask,
            need_weights=False,
        )

        mol_tokens = self.fusion_norm_mol(mol_tokens + mol_context)
        solvent_tokens = self.fusion_norm_solvent(solvent_tokens + solvent_context)

        mol_cls = mol_tokens[:, 0, :]
        solvent_cls = solvent_tokens[:, 0, :]
        solvent_desc_emb = self.solvent_desc_proj(solvent_desc)
        fused = torch.cat(
            [
                mol_cls,
                solvent_cls,
                torch.abs(mol_cls - solvent_cls),
                mol_cls * solvent_cls,
                mol_context[:, 0, :],
                solvent_context[:, 0, :],
                solvent_desc_emb,
            ],
            dim=-1,
        )
        pred = torch.cat([head(fused) for head in self.reg_heads], dim=-1)

        if eval:
            return pred
        return masked_regression_loss(pred, value, target_mask)


def masked_regression_loss(pred, target, mask, loss_type='mse'):
    mask = mask.float()
    if loss_type == 'huber':
        loss = nn.functional.smooth_l1_loss(pred, target, reduction='none')
    else:
        loss = (pred - target) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def load_compatible_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    msg = model.load_state_dict(compatible, strict=False)
    return msg, skipped


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler,
          max_length_smiles, max_length_solvent, loss_type='mse'):
    model.train()

    header = f'Train Epoch: [{epoch}]'
    print_freq = 20
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (smiles, solvents, solvent_desc, value, target_mask) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        value = value.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        solvent_desc = solvent_desc.to(device, non_blocking=True)
        smiles_input, solvent_input = tokenize_pair(
            tokenizer,
            smiles,
            solvents,
            device,
            max_length_smiles,
            max_length_solvent,
        )

        pred = model(
            smiles_input.input_ids[:, 1:],
            smiles_input.attention_mask[:, 1:],
            solvent_input.input_ids[:, 1:],
            solvent_input.attention_mask[:, 1:],
            solvent_desc,
            eval=True,
        )
        loss = masked_regression_loss(pred, value, target_mask, loss_type=loss_type)
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
    masks = []
    for smiles, solvents, solvent_desc, value, target_mask in data_loader:
        value = value.to(device, non_blocking=True)
        solvent_desc = solvent_desc.to(device, non_blocking=True)
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
            solvent_desc,
            eval=True,
        )

        preds.append(prediction.cpu())
        answers.append(value.cpu())
        masks.append(target_mask.cpu().bool())

    preds = torch.cat(preds, dim=0)
    answers = torch.cat(answers, dim=0)
    masks = torch.cat(masks, dim=0)

    norm_metrics = {}
    per_target_norm_rmse = []
    for target_idx, target in enumerate(SMILESDataset_FluoDB_MultiTask.target_names):
        target_mask = masks[:, target_idx]
        if target_mask.sum().item() == 0:
            norm_metrics[target] = float('nan')
            continue
        norm_rmse = torch.sqrt(nn.MSELoss()(preds[target_mask, target_idx], answers[target_mask, target_idx])).item()
        norm_metrics[target] = norm_rmse
        per_target_norm_rmse.append(norm_rmse)

    value_mean, value_std = denormalize
    preds = preds * value_std + value_mean
    answers = answers * value_std + value_mean
    preds = SMILESDataset_FluoDB_MultiTask.inverse_transform_targets(preds)
    answers = SMILESDataset_FluoDB_MultiTask.inverse_transform_targets(answers)

    metrics = {}
    per_target_rmse = []
    for target_idx, target in enumerate(SMILESDataset_FluoDB_MultiTask.target_names):
        target_mask = masks[:, target_idx]
        if target_mask.sum().item() == 0:
            metrics[target] = {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'n': 0}
            continue

        target_preds = preds[target_mask, target_idx]
        target_answers = answers[target_mask, target_idx]
        rmse = torch.sqrt(nn.MSELoss()(target_preds, target_answers)).item()
        mae = torch.mean(torch.abs(target_preds - target_answers)).item()
        r2 = r2_score(target_answers.numpy(), target_preds.numpy()) if target_answers.numel() > 1 else float('nan')
        metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'n': int(target_mask.sum().item())}
        per_target_rmse.append(rmse)

    metrics['mean_rmse'] = float(np.mean(per_target_rmse)) if per_target_rmse else float('inf')
    metrics['mean_norm_rmse'] = float(np.mean(per_target_norm_rmse)) if per_target_norm_rmse else float('inf')
    metrics['norm_rmse'] = norm_metrics
    return metrics


def print_metrics(prefix, metrics):
    parts = [
        f'{prefix} mean_norm_RMSE: {metrics["mean_norm_rmse"]:.4f}',
        f'mean_RMSE: {metrics["mean_rmse"]:.4f}',
    ]
    for target in SMILESDataset_FluoDB_MultiTask.target_names:
        target_metrics = metrics[target]
        parts.append(
            f'{target} RMSE: {target_metrics["rmse"]:.4f} '
            f'MAE: {target_metrics["mae"]:.4f} '
            f'R2: {target_metrics["r2"]:.4f} '
            f'norm_RMSE: {metrics["norm_rmse"][target]:.4f} '
            f'n={target_metrics["n"]}'
        )
    print(' | '.join(parts))


def resolve_log_file(args):
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(args.output_dir) / f'multitask_train_{timestamp}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def main(args, config):
    device = torch.device(args.device)
    print('DATA DIR:', args.data_dir)
    print('MULTITASK TARGETS:', ', '.join(SMILESDataset_FluoDB_MultiTask.target_names))
    print('TARGET TRANSFORMS:', SMILESDataset_FluoDB_MultiTask.target_transform_config)

    dataset_train = SMILESDataset_FluoDB_MultiTask(
        args.data_dir,
        split='train',
        solvent_descriptor_file=args.solvent_descriptor_file,
        shuffle=True,
    )
    dataset_val = SMILESDataset_FluoDB_MultiTask(
        args.data_dir,
        split='valid',
        value_mean=dataset_train.value_mean,
        value_std=dataset_train.value_std,
        solvent_desc_mean=dataset_train.solvent_desc_mean,
        solvent_desc_std=dataset_train.solvent_desc_std,
        solvent_descriptor_file=args.solvent_descriptor_file,
    )
    dataset_test = SMILESDataset_FluoDB_MultiTask(
        args.data_dir,
        split='test',
        value_mean=dataset_train.value_mean,
        value_std=dataset_train.value_std,
        solvent_desc_mean=dataset_train.solvent_desc_mean,
        solvent_desc_std=dataset_train.solvent_desc_std,
        solvent_descriptor_file=args.solvent_descriptor_file,
    )

    print(len(dataset_train), len(dataset_val), len(dataset_test))
    print('solvent descriptor dim:', dataset_train.solvent_desc_dim)

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

    print('Creating multitask model')
    model = SPMM_fluodb_multitask_regressor(
        config=config,
        tokenizer=tokenizer,
        solvent_desc_dim=dataset_train.solvent_desc_dim,
        n_targets=len(SMILESDataset_FluoDB_MultiTask.target_names),
    )
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        print('LOADING COMPATIBLE PRETRAINED WEIGHTS..')
        msg, skipped = load_compatible_checkpoint(model, args.checkpoint)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
        print('skipped incompatible keys:', len(skipped))

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
            loss_type=args.loss,
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
        print_metrics('VALID', val_stats)
        print_metrics('TEST ', test_stats)

        if val_stats['mean_norm_rmse'] < best_valid:
            best_valid = val_stats['mean_norm_rmse']
            best_test = test_stats
            save_obj = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'task': 'multitask',
                'target_names': SMILESDataset_FluoDB_MultiTask.target_names,
                'value_mean': dataset_train.value_mean,
                'value_std': dataset_train.value_std,
                'target_transform_config': SMILESDataset_FluoDB_MultiTask.target_transform_config,
                'solvent_desc_mean': dataset_train.solvent_desc_mean,
                'solvent_desc_std': dataset_train.solvent_desc_std,
                'solvent_desc_dim': dataset_train.solvent_desc_dim,
            }
            save_path = os.path.join(args.output_dir, 'multitask_best.pth')
            torch.save(save_obj, save_path)
            print('SAVED:', save_path)

        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if best_test is not None:
        print('Best test metrics from the checkpoint with best validation mean normalized RMSE:')
        print_metrics('BEST_TEST', best_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./FlourDB')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--output_dir', default='./output/FluoDB')
    parser.add_argument('--solvent_descriptor_file', default='', help='Optional CSV with solvent and physical descriptor columns.')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--min_lr', default=3e-6, type=float)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_length_smiles', default=160, type=int)
    parser.add_argument('--max_length_solvent', default=64, type=int)
    parser.add_argument('--fusion_heads', default=8, type=int)
    parser.add_argument('--solvent_desc_width', default=128, type=int)
    parser.add_argument('--loss', default='mse', choices=['mse', 'huber'])
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--log_file', default='', help='Path to save the training log. Defaults to output_dir/multitask_train_TIMESTAMP.log.')
    args = parser.parse_args()

    cls_config = {
        'batch_size_train': args.batch_size,
        'batch_size_test': 64,
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'fusion_heads': args.fusion_heads,
        'fusion_dropout': 0.1,
        'solvent_desc_width': args.solvent_desc_width,
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
    log_path = resolve_log_file(args)
    with open(log_path, 'a', encoding='utf-8', buffering=1) as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print('LOG FILE:', log_path)
            main(args, cls_config)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

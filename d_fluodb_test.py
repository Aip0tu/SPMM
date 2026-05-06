import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SMILESDataset_FluoDB
from d_fluodb_regression import (
    SPMM_fluodb_regressor,
    build_tokenizer,
    evaluate,
    resolve_dataset,
)


def default_config():
    return {
        'batch_size_test': 64,
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
    }


def resolve_checkpoint(args):
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.output_dir, f'{args.task}_best.pth')
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'{checkpoint_path} does not exist. Train first or pass --checkpoint explicitly.'
        )
    return checkpoint_path


def resolve_device(device_name):
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no CUDA GPU is available. Use --device cpu.')
    return torch.device(device_name)


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model')
        if state_dict is not None:
            return state_dict
    return checkpoint


def strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def build_test_dataset(args, checkpoint):
    train_path, _, test_path = resolve_dataset(args.task, args.data_dir)
    value_mean = checkpoint.get('value_mean') if isinstance(checkpoint, dict) else None
    value_std = checkpoint.get('value_std') if isinstance(checkpoint, dict) else None

    if value_mean is None or value_std is None:
        dataset_train = SMILESDataset_FluoDB(train_path, target_name=args.task)
        value_mean = dataset_train.value_mean
        value_std = dataset_train.value_std
        print('Checkpoint has no target scaler. Recomputed scaler from train split.')

    dataset_test = SMILESDataset_FluoDB(
        test_path,
        target_name=args.task,
        value_mean=value_mean,
        value_std=value_std,
    )
    return dataset_test, (value_mean, value_std)


def main(args):
    checkpoint_path = resolve_checkpoint(args)
    device = resolve_device(args.device)

    print('TASK:', args.task)
    print('DATA DIR:', args.data_dir)
    print('CHECKPOINT:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', default_config()) if isinstance(checkpoint, dict) else default_config()
    tokenizer = build_tokenizer(args.vocab_filename)

    dataset_test, denormalize = build_test_dataset(args, checkpoint)
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = SPMM_fluodb_regressor(config=config, tokenizer=tokenizer)
    state_dict = strip_module_prefix(get_state_dict(checkpoint))
    load_msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % checkpoint_path)
    print(load_msg)
    if any(key.startswith('reg_head') for key in load_msg.missing_keys):
        print('WARNING: regression head weights are missing. Use a fine-tuned FluoDB checkpoint.')

    model = model.to(device)
    test_stats = evaluate(
        model,
        test_loader,
        tokenizer,
        device,
        denormalize,
        args.max_length_smiles,
        args.max_length_solvent,
    )
    print('TEST SAMPLES:', len(dataset_test))
    print('TEST RMSE: %.4f | MAE: %.4f | R2: %.4f' % (
        test_stats['rmse'],
        test_stats['mae'],
        test_stats['r2'],
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='emi', choices=['abs', 'emi', 'plqy', 'e'])
    parser.add_argument('--data_dir', default='./FlourDB')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output_dir', default='./output/FluoDB')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_length_smiles', default=128, type=int)
    parser.add_argument('--max_length_solvent', default=64, type=int)
    main(parser.parse_args())

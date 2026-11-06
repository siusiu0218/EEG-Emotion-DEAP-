"""Allocated-node smoke check: actual DEAP windows, GTENet, and CLI resume."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    assert source['epoch'] == 47
    assert all(len(v) == 48 and np.isfinite(v).all() for v in source['history'].values())
    assert abs(source['best_val_acc'] - 0.6492838541666667) < 1e-12
    script = Path(__file__).with_name('GTE_train.py')
    common = [sys.executable, '-u', str(script), '--dataset', 'DEAP',
              '--data_path', 'dataset/DEAP/data_preprocessed_python',
              '--preprocessed_data_path', 'dataset/DEAP/data_time_window_1s',
              '--subject_ids', '1', '--max_samples', '32', '--batch_size', '32',
              '--lr', '1e-4', '--weight_decay', '1e-4', '--time_window', '1',
              '--gat_heads', '2', '--transformer_heads', '4', '--transformer_layers', '1',
              '--d_model', '128', '--ffn_dim', '512', '--data_augmentation', '0.3',
              '--early_stopping_patience', '100']
    subprocess.run(common + ['--epochs', '1', '--model_save_path', str(root / 'initial' / 'best.pth')], check=True)
    initial = torch.load(root / 'initial' / 'best_last.pth', weights_only=True)
    copied = root / 'arbitrary_parent_with_underscores' / 'source.pth'
    copied.parent.mkdir()
    shutil.copy2(root / 'initial' / 'best_last.pth', copied)
    before = hashlib.sha256(copied.read_bytes()).hexdigest()
    subprocess.run(common + ['--epochs', '2', '--resume_from', str(copied),
                            '--model_save_path', str(root / 'resumed' / 'best.pth')], check=True)
    last = torch.load(root / 'resumed' / 'best_last.pth', weights_only=True)
    assert last['epoch'] == 1
    for key, values in last['history'].items():
        assert len(values) == 2 and np.isfinite(values).all()
        assert values[0] == initial['history'][key][0]
    assert all(state['step'].item() == 2 for state in last['optimizer_state_dict']['state'].values())
    assert last['scheduler_state_dict']['last_epoch'] == 2
    assert last['config']['data_augmentation'] == 0.3
    assert any(not torch.equal(v, initial['model_state_dict'][k]) for k, v in last['model_state_dict'].items())
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == before
    assert list((root / 'resumed').glob('*_final_curves.png'))
    print('REAL_RESUME_PASSED: 32 real windows (26 train/6 val), epoch 1 -> 2, state restored; source epoch 48 confirmed', flush=True)


if __name__ == '__main__':
    main()

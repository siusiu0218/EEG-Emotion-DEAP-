"""Reload a small-run checkpoint and independently verify its validation accuracy."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from GTE_train import DEAPDataset
from model_optimized import GTENet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--full', action='store_true', help='Verify all 32 subjects instead of the 320-window smoke subset.')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    checkpoints = list(run_dir.glob('*_train_acc_*.pth'))
    assert len(checkpoints) == 1, checkpoints
    checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 0
    assert checkpoint['optimizer_state_dict']['state'], 'No optimizer updates saved'
    for values in checkpoint['history'].values():
        assert len(values) == 1 and np.isfinite(values).all(), values
    curves = list(run_dir.glob('*_final_curves.png'))
    assert len(curves) == 1 and curves[0].stat().st_size > 0
    dataset = DEAPDataset('dataset/DEAP/data_preprocessed_python', 1,
                          preprocessed_data_path='dataset/DEAP/data_time_window_1s', subject_ids=None if args.full else [1])
    if args.full:
        assert len(dataset) == 76800
        subset = dataset
    else:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:320]
        subset = Subset(dataset, indices.tolist())
    val_size = int(len(subset) * 0.2)
    _, validation = random_split(subset, [len(subset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    model = GTENet(n_channels=32, n_freq_bands=5, time_length=128, num_classes=2,
                   gat_heads=2, transformer_heads=4, transformer_layers=1, d_model=128, ffn_dim=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    predictions, labels_all = [], []
    with torch.no_grad():
        for time, freq, labels in DataLoader(validation, batch_size=32):
            assert torch.isfinite(time).all() and torch.isfinite(freq).all()
            logits = model(time, freq)
            assert torch.isfinite(logits).all()
            predictions.extend(logits.argmax(1).tolist())
            labels_all.extend(labels.tolist())
    accuracy = float(np.mean(np.array(predictions) == labels_all))
    assert abs(accuracy - checkpoint['best_val_acc']) < 1e-12
    assert abs(accuracy - checkpoint['history']['val_acc'][0]) < 1e-12
    print(f'OK: checkpoint reloaded, optimizer updated, finite history, curve saved; val_n={len(validation)} accuracy={accuracy:.8f}')


if __name__ == '__main__':
    main()

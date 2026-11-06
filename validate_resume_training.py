"""Fast regression of the actual train() resume path; no research data required."""
import argparse
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

import GTE_train as training


class TinyModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, time, freq):
        return self.classifier(time)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dataset = TensorDataset(torch.randn(10, 2), torch.randn(10, 2), torch.arange(10) % 2)
    with tempfile.TemporaryDirectory(prefix='gte_resume_') as directory:
        root = Path(directory)
        args = argparse.Namespace(
            dataset='DEAP', data_path='unused', preprocessed_data_path='unused',
            subject_ids=None, max_samples=None, time_window=1, batch_size=32,
            gat_heads=2, transformer_heads=4, transformer_layers=1, d_model=128,
            ffn_dim=512, lr=1e-4, weight_decay=1e-4, epochs=1,
            data_augmentation=0.0, early_stopping_patience=100, resume_from=None,
            model_save_path=str(root / 'initial' / 'best.pth'))
        with patch.object(training, 'DEAPDataset', return_value=dataset), \
             patch.object(training, 'GTENet', TinyModel), \
             patch.object(training, 'plot_curves'), \
             patch.object(training.torch.cuda, 'is_available', return_value=False), \
             patch.object(training, 'augment_eeg_advanced', side_effect=AssertionError('Augmentation changed')):
            training.train(args)
            original = next((root / 'initial').glob('*_train_acc_*.pth'))
            checkpoint = torch.load(original, weights_only=True)
            # Deliberately arbitrary name: metadata must come from the checkpoint.
            source = root / 'arbitrary_parent_with_underscores' / 'checkpoint.pth'
            source.parent.mkdir()
            checkpoint['best_val_acc'] = 1.0  # Also exercise no new best after resuming.
            checkpoint['scheduler_state_dict']['best'] = 0.0
            checkpoint['scheduler_state_dict']['num_bad_epochs'] = 3
            torch.save(checkpoint, source)
            before = source.read_bytes()
            args.resume_from = str(source)
            args.epochs = 2
            args.model_save_path = str(root / 'resumed' / 'best.pth')
            training.train(args)
            last = torch.load(root / 'resumed' / 'best_last.pth', weights_only=True)
            assert last['epoch'] == 1, last['epoch']
            assert all(len(values) == 2 for values in last['history'].values())
            assert last['best_val_acc'] == 1.0
            for state in last['optimizer_state_dict']['state'].values():
                assert state['step'].item() == 2, state['step']
            assert last['scheduler_state_dict']['num_bad_epochs'] == 4
            assert last['config']['data_augmentation'] == 0.0
            assert source.read_bytes() == before, 'Source checkpoint changed'
            assert (root / 'resumed' / 'best_resume_source.pth').is_file(), 'Resume source must be kept'
            missing = copy.copy(args)
            missing.resume_from = str(root / 'missing.pth')
            try:
                training.train(missing)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing resume must not silently start over')
    print('RESUME_REGRESSION_PASSED: epoch/history/optimizer/scheduler/config/source/no-new-best/missing-file')


if __name__ == '__main__':
    main()

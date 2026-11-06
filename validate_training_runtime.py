"""Regression checks for scheduler compatibility and both time-warp branches."""
import ast
from pathlib import Path
from unittest.mock import patch

import torch

from GTE_train import augment_eeg_advanced


def main():
    print(f'Runtime PyTorch: {torch.__version__}', flush=True)
    tree = ast.parse(Path(__file__).with_name('GTE_train.py').read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'ReduceLROnPlateau']
    assert len(calls) == 2
    for call in calls:
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        scheduler.step(0.7)
    x = torch.arange(2 * 32 * 128, dtype=torch.float32).reshape(2, 32, 128)
    for scale in (0.9, 1.1):
        with patch('GTE_train.random.random', side_effect=[1.0, 0.0, 1.0]), \
             patch('GTE_train.random.uniform', return_value=scale):
            y = augment_eeg_advanced(x, 'cpu')
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert torch.equal(x, torch.arange(x.numel(), dtype=x.dtype).reshape_as(x))
    print('OK: both scheduler call sites; time warp shrink/expand; input preserved')


if __name__ == '__main__':
    main()

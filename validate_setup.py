"""Fast, read-only checks before preprocessing or training GTE-Net."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from model_optimized import GTENet
from DatasetProcessSave import calculate_ae, calculate_de, stimulus_window_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/DEAP/data_preprocessed_python"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_files = sorted(args.data_dir.glob("s*.dat"))
    if len(subject_files) != 32:
        raise RuntimeError(f"Expected 32 DEAP subject files, found {len(subject_files)}")

    with subject_files[0].open("rb") as handle:
        subject = pickle.load(handle, encoding="latin1")

    data = np.asarray(subject["data"])
    labels = np.asarray(subject["labels"])
    if data.shape != (40, 40, 8064):
        raise RuntimeError(f"Unexpected DEAP data shape: {data.shape}")
    if labels.shape != (40, 4):
        raise RuntimeError(f"Unexpected DEAP label shape: {labels.shape}")
    starts = list(stimulus_window_starts(data.shape[-1], 128, 128))
    if len(starts) != 60 or starts[0] != 384 or starts[-1] != 7936:
        raise RuntimeError(f"Unexpected DEAP stimulus windows: {starts[:1]}...{starts[-1:]}")

    eeg = data[0, :32]
    baseline = calculate_ae(eeg[:, :384], 128)
    first_window = eeg[:, starts[0] : starts[0] + 128] - baseline
    freq_features = np.stack(
        [calculate_de(channel, dataset_type="DEAP") for channel in first_window]
    )
    if first_window.shape != (32, 128) or freq_features.shape != (32, 5):
        raise RuntimeError(
            f"Unexpected feature shapes: time={first_window.shape}, freq={freq_features.shape}"
        )
    if not np.isfinite(freq_features).all():
        raise RuntimeError("DE features contain non-finite values")

    model = GTENet(
        n_channels=32,
        n_freq_bands=5,
        time_length=128,
        num_classes=2,
        gat_heads=2,
        transformer_heads=4,
        transformer_layers=1,
        d_model=128,
    ).eval()
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(first_window).float().unsqueeze(0),
            torch.from_numpy(freq_features).float().unsqueeze(0),
        )
    if logits.shape != (1, 2):
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

    print(f"OK: {len(subject_files)} DEAP files")
    print(f"OK: data={data.shape}, labels={labels.shape}")
    print("OK: 60 one-second stimulus windows, baseline excluded")
    print(f"OK: real EEG features time={first_window.shape}, freq={freq_features.shape}")
    print(f"OK: model forward output={tuple(logits.shape)}")
    print(f"PyTorch device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")


if __name__ == "__main__":
    main()

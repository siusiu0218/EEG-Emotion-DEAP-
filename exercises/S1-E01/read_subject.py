"""S1-E01：补全 load_subject，读取 EEG 并生成 Valence 二分类标签。"""

import pickle
from pathlib import Path

import numpy as np


def load_subject(file_path):
    """返回 eeg（40,32,8064）和 targets（40,），targets 为整数 0/1。"""
    # 文件读取方式已提供；数据选取和标签转换由你实现。
    with open(file_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    # TODO 1：从 subject['data'] 中选出全部试验的前 32 个通道。
    # TODO 2：从 subject['labels'] 中选出每次试验的 Valence（第 0 列）。
    # TODO 3：评分 < 5 为 0，>= 5 为 1；返回两个 NumPy 数组。
    raise NotImplementedError("请补全数据选取与标签转换，再删除这一行")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "dataset/DEAP/data_preprocessed_python/s01.dat"
    eeg, targets = load_subject(file_path)

    print("EEG shape:", eeg.shape)
    print("Targets shape:", targets.shape)
    print("Targets dtype:", targets.dtype)
    print("First 10 targets:", targets[:10])

    assert eeg.shape == (40, 32, 8064), "检查试验、通道和采样点的选择"
    assert targets.shape == (40,), "每次试验应当只有一个类别标签"
    assert np.issubdtype(targets.dtype, np.integer), "标签应为整数数组"
    assert np.isin(targets, [0, 1]).all(), "标签只能包含 0 和 1"
    print("基本形状与类型检查通过；标签对应关系仍需评审。")

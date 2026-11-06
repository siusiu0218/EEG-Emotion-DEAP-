import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal
import torch
def calculate_de(signal_window,dataset_type):
    """
    计算一个信号窗口在不同频带上的差分熵 (DE)。

    计算流程：
    1.  将信号窗口通过带通滤波器组（如Delta, Theta, Alpha, Beta, Gamma）。
    2.  对每个频带的信号，计算其差分熵。对于高斯分布的信号，DE = 0.5 * log(2 * pi * e * variance)。

    输入:
        signal_window (np.array): 单个通道的EEG时间序列片段，形状 (T_slice,)
    输出:
        de_features (np.array): 5个频带的DE特征，形状 (5,)
    """
    fs = 200
    de_features = []
    if dataset_type =='SEED':
        fs = 200
    elif dataset_type =='DEAP':
        fs = 128  # DEAP Python 预处理版采样率
    variances = []
    bands = [[0.5,4.],[4.,8.],[8.,13.],[13.,30.],[30.,50.]]
    for low, high in bands:
        b, a = signal.butter(8, [low / (fs / 2), high / (fs / 2)], btype='bandpass')
        filtered = signal.filtfilt(b, a, signal_window)
        var = np.var(filtered)
        # 添加安全检查
        if var <= 0:
            var = 1e-8  # 防止方差为0或负数
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        if np.isnan(de) or np.isinf(de):
            de = 0.0
        de_features.append(de)

    return np.array(de_features)


# 预计算滤波器系数（避免重复计算）
def _precompute_filter_coefficients(fs, bands):
    """预计算所有频带的滤波器系数"""
    filters = {}
    for i, (low, high) in enumerate(bands):
        b, a = signal.butter(8, [low / (fs / 2), high / (fs / 2)], btype='bandpass')
        filters[i] = (b, a)
    return filters


# 预定义滤波器系数
DEAP_FILTERS = _precompute_filter_coefficients(128, [[0.5, 4.], [4., 8.], [8., 13.], [13., 30.], [30., 50.]])
SEED_FILTERS = _precompute_filter_coefficients(200, [[0.5, 4.], [4., 8.], [8., 13.], [13., 30.], [30., 50.]])


def calculate_de_fast(signal_window, dataset_type):
    """
    优化版的DE计算函数
    """
    if dataset_type == 'SEED':
        fs = 200
        filters = SEED_FILTERS
    else:  # DEAP
        fs = 128
        filters = DEAP_FILTERS

    de_features = np.zeros(5)
    bands = [[0.5, 4.], [4., 8.], [8., 13.], [13., 30.], [30., 50.]]

    # 使用预计算的滤波器系数和lfilter（比filtfilt快）
    for i, (low, high) in enumerate(bands):
        b, a = filters[i]
        filtered = signal.lfilter(b, a, signal_window)  # 比filtfilt快

        var = np.var(filtered, ddof=0)
        if var <= 0:
            var = 1e-8

        de = 0.5 * np.log(2 * np.pi * np.e * var)
        if np.isnan(de) or np.isinf(de):
            de = 0.0

        de_features[i] = de

    return de_features


def calculate_de_batch(signal_windows, dataset_type):
    """
    批量计算DE特征，显著提高效率
    输入: signal_windows (np.array) 形状为 (N, T)
    输出: de_features (np.array) 形状为 (N, 5)
    """
    if dataset_type == 'SEED':
        fs = 200
        filters = SEED_FILTERS
    else:  # DEAP
        fs = 128
        filters = DEAP_FILTERS

    n_signals = signal_windows.shape[0]
    de_features = np.zeros((n_signals, 5))

    # 对每个频带批量处理
    for i, (low, high) in enumerate([[0.5, 4.], [4., 8.], [8., 13.], [13., 30.], [30., 50.]]):
        b, a = filters[i]

        # 批量滤波
        filtered = signal.lfilter(b, a, signal_windows, axis=1)

        # 批量计算方差
        var = np.var(filtered, axis=1, ddof=0)

        # 数值稳定性
        var[var <= 0] = 1e-8

        # 批量计算DE
        de = 0.5 * np.log(2 * np.pi * np.e * var)

        # 处理数值异常
        de = np.nan_to_num(de, nan=0.0, posinf=0.0, neginf=0.0)

        de_features[:, i] = de

    return de_features

# -------------------------
# 2. PyTorch 数据集类
# -------------------------
def calculate_ae(basic_eeg_signal,time_window_size):
    # 输入一个基线脑电信号和时间窗口大小，返回的是一个时间窗口大小的平均基线信号
    n_slices = basic_eeg_signal.shape[1]//time_window_size
    average_eeg = np.zeros((basic_eeg_signal.shape[0], time_window_size))
    for i in range(n_slices):
        start = i*time_window_size
        end = (i+1)*time_window_size
        average_eeg += basic_eeg_signal[:,start:end]
    average_eeg = average_eeg/n_slices
    return average_eeg


def stimulus_window_starts(total_points, time_window_length, frequence=128):
    """Return non-overlapping window starts after DEAP's three-second baseline."""
    stimulus_start = 3 * frequence
    return range(
        stimulus_start,
        total_points - time_window_length + 1,
        time_window_length,
    )

def save_preprocessed_data(data_path, save_path, time_window_length, frequence=128, n_channels=32):
    """
        预处理DEAP数据并保存，避免每次训练重复处理
        Args:
            data_path: 原始DEAP数据路径
            save_path: 预处理数据保存路径
            time_window_length: 时间窗口长度
            frequence: 采样频率
            n_channels: 通道数
        输出可得到每一个数据集文件处理之后的数据集。
        """
    print("Preprocessing and saving DEAP dataset...")
    os.makedirs(save_path, exist_ok=True)
    subject_files = sorted(f for f in os.listdir(data_path) if f.endswith('.dat'))
    for subject_idx, subject_file in enumerate(tqdm(subject_files, desc="Processing Subjects")):
        # TODO: 用户需要根据自己的文件路径实现完整的数据加载逻辑
        file_path = os.path.join(data_path, subject_file)
        subject_samples_time = []
        subject_samples_freq = []
        subject_labels = []
        # 使用pickle加载.dat文件
        with open(file_path, 'rb') as f:
            subject_data = pickle.load(f, encoding='latin1')

        raw_eeg_data = subject_data['data'][:, :32, :]  # 形状 (40, 40, 8064)，即 (试验, 通道, 时间点)改成（40，32，8064）
        raw_labels = subject_data['labels']  # 形状 (40, 4)，即 (试验, Valence/Arousal/Dominance/Liking)

        # 对每个试验进行处理
        for i in range(raw_eeg_data.shape[0]):
            trial_eeg = raw_eeg_data[i]  # 一个eeg信号对应一个标签,[32,8064]
            trial_labels = raw_labels[i]
            # 我们这里只做Valence的二分类，以5为阈值，为什么二分类？
            valence_label = 1 if trial_labels[0] >= 5.0 else 0
            # 数据预处理，计算平均基线信号
            basic_eeg_signal = trial_eeg[:, :3 * frequence]  # DEAP数据集基线信号是前三秒
            average_eeg_signal = calculate_ae(basic_eeg_signal, time_window_length)
            for start_point in stimulus_window_starts(
                trial_eeg.shape[1], time_window_length, frequence
            ):
                # 跳过前三秒基线，只对后60秒刺激信号切窗。
                end_point = start_point + time_window_length
                eeg_window = trial_eeg[:, start_point:end_point]  # -> (32,起始点到终止点)得到一个3s的窗口
                eeg_window = eeg_window - average_eeg_signal
                # --- 特征计算 ---
                # 1. 时域信号 (x_time)
                x_time = eeg_window
                # 2. 频域特征 (x_freq - DE)
                # 我们需要对整个窗口计算DE特征
                x_freq = np.zeros((n_channels, 5))
                for ch in range(n_channels):  # 计算每一个通道的五个频域的特征
                    x_freq[ch, :] = calculate_de(eeg_window[ch, :], dataset_type='DEAP')
                # --- 标签处理 ---
                # 将输入对应标签。
                subject_samples_time.append(x_time)
                subject_samples_freq.append(x_freq)
                subject_labels.append(valence_label)
        # 走到此处说明一个.mat文件的数据被处理完成，此时要生成一个文件
        one_subject_data = {
            'samples_time': subject_samples_time,
            'samples_freq': subject_samples_freq,
            'labels': subject_labels,
        }
        subject_id = os.path.splitext(subject_file)[0].removeprefix('s')
        one_subject_path = os.path.join(save_path, f'subject_{int(subject_id):02d}.npz')
        np.savez(one_subject_path, **one_subject_data)
    print(f"Preprocessing completed! Saved to: {save_path}")
    return save_path


def load_preprocessed_data(load_path, subject_ids=None):
    """
    加载预处理好的DEAP数据

    Args:
        load_path: 预处理数据路径
        subject_ids: 要加载的subject ID列表，None表示加载所有

    Returns:
        samples: 包含'time'和'freq'的字典列表
        labels: 标签列表
        dataset_info: 数据集信息
    """
    print("Loading preprocessed DEAP dataset...")
    samples = []
    labels = []

    if subject_ids is None:
        # 加载所有subject
        subject_files = [f for f in os.listdir(load_path) if f.startswith('subject_') and f.endswith('.npz')]
    else:
        # 加载指定subject
        subject_files = [f'subject_{sid:02d}.npz' for sid in subject_ids]

    for subject_file in tqdm(subject_files, desc="Loading Subjects"):
        file_path = os.path.join(load_path, subject_file)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        subject_data = np.load(file_path, allow_pickle=True)

        subject_samples_time = subject_data['samples_time']
        subject_samples_freq = subject_data['samples_freq']
        subject_labels = subject_data['labels']

        # 转换为字典格式
        for i in range(len(subject_labels)):
            samples.append({
                'time': subject_samples_time[i],
                'freq': subject_samples_freq[i]
            })
            labels.append(subject_labels[i])

    print(f"Loaded {len(samples)} samples from {len(subject_files)} subjects")

    return samples, labels


def check_preprocessed_data_exists(load_path,data_path):
    """
    检查预处理数据是否存在

    Args:
        load_path: 预处理数据路径

    Returns:
        bool: 是否存在完整的预处理数据
    """
    if not os.path.exists(load_path):
        return False
    subject_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
    # 检查是否有至少一个subject文件
    pre_subject_files = [f for f in os.listdir(load_path) if f.startswith('subject_') and f.endswith('.npz')]
    return len(subject_files) == len(pre_subject_files)
def interaug_with_freq_optimized(timg, freq, label, device, dataset_type='DEAP'):
    """
    优化版的跨样本片段重组增强
    """
    batch_size, n_channels, time_points = timg.shape
    segment_length = time_points // 8

    aug_data = []
    aug_label = []
    aug_freq = []

    for cls4aug in range(2):
        cls_mask = (label == cls4aug)
        cls_indices = torch.where(cls_mask)[0]

        if len(cls_indices) == 0:
            continue

        tmp_data = timg[cls_indices]
        tmp_label = label[cls_indices]
        num_aug_per_class = min(batch_size // 2, len(tmp_data))

        if num_aug_per_class <= 0:
            continue

        tmp_aug_data = torch.zeros((num_aug_per_class, n_channels, time_points), device=device)

        # 生成增强样本
        for ri in range(num_aug_per_class):
            for rj in range(8):
                if len(tmp_data) > 0:
                    rand_idx = torch.randint(0, len(tmp_data), (8,), device=device)
                    segment_start = rj * segment_length
                    segment_end = (rj + 1) * segment_length

                    tmp_aug_data[ri, :, segment_start:segment_end] = \
                        tmp_data[rand_idx[rj], :, segment_start:segment_end]

        # 批量计算频域特征（关键优化）
        tmp_aug_data_cpu = tmp_aug_data.cpu().numpy()

        # 重塑为 (num_aug_per_class * n_channels, time_points) 进行批量计算
        batch_signals = tmp_aug_data_cpu.reshape(-1, time_points)
        batch_de_features = calculate_de_batch(batch_signals, dataset_type)

        # 重塑回原始形状
        tmp_aug_freq = batch_de_features.reshape(num_aug_per_class, n_channels, 5)
        tmp_aug_freq = torch.from_numpy(tmp_aug_freq).float().to(device)

        aug_data.append(tmp_aug_data)
        aug_freq.append(tmp_aug_freq)
        aug_label.append(tmp_label[:num_aug_per_class])

    if not aug_data:
        return timg, freq, label

    aug_data = torch.cat(aug_data, dim=0)
    aug_freq = torch.cat(aug_freq, dim=0)
    aug_label = torch.cat(aug_label, dim=0)

    shuffle_idx = torch.randperm(len(aug_data))
    aug_data = aug_data[shuffle_idx]
    aug_freq = aug_freq[shuffle_idx]
    aug_label = aug_label[shuffle_idx]

    return aug_data, aug_freq, aug_label

def interaug_with_freq(timg, freq,label,device,dataset_type='DEAP'): # 要根据输入数据进行修改。输入数据太短分割与重建性能下降很厉害。
    """
    完整的跨样本片段重组增强，同时处理时域和频域特征

    Args:
        timg: 时域数据 (B, C, T)
        label: 标签 (B,)
        dataset_type: 数据集类型
    """
    aug_data = []
    aug_label = []
    aug_freq = []
    batch_size = timg.shape[0]
    segment_length = timg.shape[2]/2
    n_channels = timg.shape[1]
    # 对每个类别分别处理
    for cls4aug in range(2):
        cls_idx = np.where(label.cpu().numpy() == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        num_aug_per_class = min(batch_size // 2, len(tmp_data))
        tmp_aug_data = torch.zeros((num_aug_per_class, n_channels, timg.shape[2]),device=device)
        for ri in range(num_aug_per_class):
            for rj in range(2):  # 8个片段
                rand_idx = torch.randint(0, len(tmp_data), (2,), device=device)
                segment_start = rj * int(segment_length)
                segment_end = (rj + 1) * int(segment_length)

                # 重组时域信号
                tmp_aug_data[ri, :, segment_start:segment_end] = \
                    tmp_data[rand_idx[rj], :, segment_start:segment_end]
        # 将数据移动到CPU并转换为numpy进行计算
        tmp_aug_freq = torch.zeros((num_aug_per_class, n_channels, 5), device=device)
        tmp_aug_data_cpu = tmp_aug_data.cpu().numpy()

        for i in range(num_aug_per_class):
            for ch in range(n_channels):
                # 计算每个通道的DE特征
                de_features = calculate_de(tmp_aug_data_cpu[i, ch, :], dataset_type)
                tmp_aug_freq[i, ch, :] = torch.from_numpy(de_features).float()

        # 将频域特征移回设备
        tmp_aug_freq = tmp_aug_freq.to(device)

        aug_data.append(tmp_aug_data)
        aug_freq.append(tmp_aug_freq)
        aug_label.append(tmp_label[:num_aug_per_class])
    if not aug_data:
        print("Warning: No augmentation data generated, returning original data")
        return timg, freq, label  # 返回原始数据和空的频域特征
    # 合并所有增强数据
    aug_data = torch.cat(aug_data, dim=0)
    aug_freq = torch.cat(aug_freq, dim=0)
    aug_label = torch.cat(aug_label, dim=0)

    # 打乱顺序
    shuffle_idx = torch.randperm(len(aug_data))
    aug_data = aug_data[shuffle_idx]
    aug_freq = aug_freq[shuffle_idx]
    aug_label = aug_label[shuffle_idx]
    return aug_data, aug_freq, aug_label

# 使用示例
if __name__ == "__main__":
    # 预处理并保存数据（只需要运行一次）
    raw_data_path = "./dataset/DEAP/data_preprocessed_python"
    preprocessed_path = "./dataset/DEAP/data_time_window_1s"

    # 第一次运行：预处理并保存
    # save_preprocessed_data(data_path=raw_data_path, save_path=preprocessed_path,time_window_length=128)

    # 后续使用：直接加载预处理数据
    # samples, labels= load_preprocessed_data(preprocessed_path)

    # 检查数据是否存在
    exists = check_preprocessed_data_exists("",raw_data_path)
    print(f"Preprocessed data exists: {exists}")

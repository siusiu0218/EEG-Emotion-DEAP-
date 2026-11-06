# train_regularized.py - 修复版本
"""
GTE-Net 模型的最终优化版训练脚本。

功能:
- 从检查点恢复训练。
- 训练结束后绘制并保存损失和准确率曲线。
- 使用数据缓存避免重复预处理。
- 包含数据标准化。
- 包含梯度裁剪。
- 使用 AdamW 优化器和权重衰减。
- 使用 ReduceLROnPlateau 学习率调度器（基于验证损失）。
- 实现早停机制。
- 可通过命令行调整 Dropout 和标签平滑。
- 增加了 NaN/Inf 检查以提高稳定性。

[修复说明]：
1. 修复了 DataLoader 多进程导致 CUDA/CPU 设备冲突的问题。
   - 将 DataLoader 的 num_workers 设置为 0。
   - 确保 Dataset 中的 self.mean 和 self.std 始终是 CPU Tensor。
2. 修复了 __getitem__ 中错误的张量创建代码。
"""
import os
import argparse
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
from scipy import signal
import math
import time  # 用于生成唯一文件名

import torch
import torch.nn as nn
# [修改] 导入 ReduceLROnPlateau 而不是 CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# 确保导入的是优化版的模型
from model_optimized import GTENet  # 假设您的优化模型在此文件中
# 导入数据处理函数
from DatasetProcessSave import save_preprocessed_data, load_preprocessed_data, check_preprocessed_data_exists, \
    calculate_de


# -------------------------
# 1. 数据集类 (集成缓存逻辑)
# -------------------------
class EEGDataset(Dataset):
    """通用EEG数据集基类，包含标准化，处理时域和频域数据。"""

    def __init__(self):
        self.samples_time = []  # 存储时域数据 (Numpy arrays)
        self.samples_freq = []  # 存储频域数据 (Numpy arrays)
        self.labels = []  # 存储标签 (integers)
        self.mean = None  # 标准化均值 (Tensor) - 必须在 CPU 上
        self.std = None  # 标准化标准差 (Tensor) - 必须在 CPU 上

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 确保索引在范围内
        if idx >= len(self.labels):
            raise IndexError("Index out of bounds")

        x_time_np = self.samples_time[idx]
        x_freq_np = self.samples_freq[idx]
        label = self.labels[idx]

        # x_time/x_freq 必须是 CPU 张量，DataLoader worker 才能处理
        x_time = torch.FloatTensor(x_time_np)
        x_freq = torch.FloatTensor(x_freq_np)

        # 应用标准化 (只对时域数据)
        if self.mean is not None and self.std is not None:
            # 这里的 self.mean/std 必须是 CPU Tensor，因为在 __getitem__ 中执行
            current_mean = self.mean.to('cpu')  # [FIX 1]: 确保统计量在 CPU
            current_std = self.std.to('cpu')  # [FIX 1]: 确保统计量在 CPU

            if current_mean.ndim == 1: current_mean = current_mean.unsqueeze(1)  # (M) -> (M, 1)
            if current_std.ndim == 1: current_std = current_std.unsqueeze(1)  # (M) -> (M, 1)

            if x_time.shape[0] == current_mean.shape[0]:  # 确保通道数匹配
                # [FIX 2]: 移除错误的张量创建和多余的标准化代码
                x_time = (x_time - current_mean) / (current_std + 1e-8)  # 加 epsilon 防止除零
            else:
                print(
                    f"Warning: Channel mismatch during normalization. Data shape {x_time.shape}, Mean shape {current_mean.shape}. Skipping normalization for this item.")

        label = torch.LongTensor([label]).squeeze()

        # x_time, x_freq, label 都是 CPU Tensor，在 Dataloader 迭代时会被移到 GPU
        return x_time, x_freq, label

    def calculate_norm_stats(self, data_indices):
        """根据训练数据索引计算均值和标准差 (只针对时域数据)"""
        if not data_indices:
            print("Warning: No indices provided for normalization calculation.")
            return
        print("Calculating normalization statistics...")
        # 从 self.samples_time 中根据索引提取训练数据
        train_data_list = [self.samples_time[i] for i in data_indices if i < len(self.samples_time)]
        if not train_data_list:
            print("Warning: Could not retrieve training data for normalization.")
            return

        all_data = np.stack(train_data_list)  # -> (N_train, M, T)
        # 沿着批次和时间维度计算每个通道的均值和标准差
        mean_np = np.mean(all_data, axis=(0, 2))  # -> (M,)
        std_np = np.std(all_data, axis=(0, 2))  # -> (M,)

        # [FIX 3]: 确保统计量被创建为 CPU Tensor
        self.mean = torch.FloatTensor(mean_np).to('cpu')
        self.std = torch.FloatTensor(std_np).to('cpu')

        # 防止标准差为0
        self.std[self.std == 0] = 1.0
        print(
            f"Normalization stats calculated: mean shape {self.mean.shape}, std shape {self.std.shape}. On device: {self.mean.device}")


class DEAPDataset(EEGDataset):
    """加载DEAP数据集, 使用缓存"""

    def __init__(self, data_path, time_window_size, frequence=128, n_channels=32, preprocessed_data_path=None):
        super().__init__()
        self.data_path = data_path
        self.time_window_length = int(time_window_size * frequence)  # 转换为样本点数
        self.frequence = frequence
        self.n_channels = n_channels

        # 检查或生成缓存
        if not preprocessed_data_path:
            preprocessed_data_path = os.path.join(os.path.dirname(data_path),
                                                  f"deap_preprocessed_tw{time_window_size}s_freq{frequence}.pkl")
            print(f"Preprocessed data path not specified, using default: {preprocessed_data_path}")

        if check_preprocessed_data_exists(load_path=preprocessed_data_path, data_path=data_path):
            self._load_preprocessed(preprocessed_data_path)
        else:
            print("Preprocessed data not found or incomplete. Generating...")
            # 调用外部函数处理并保存
            save_preprocessed_data(data_path=self.data_path,
                                   save_path=preprocessed_data_path,
                                   time_window_length=self.time_window_length,
                                   frequence=self.frequence,
                                   n_channels=self.n_channels)
            self._load_preprocessed(preprocessed_data_path)  # 处理完后加载

    def _load_preprocessed(self, path):
        print(f"Loading preprocessed DEAP data from {path}...")
        # 修改为加载 save_preprocessed_data 保存的 .npz 文件
        loaded_samples, loaded_labels = load_preprocessed_data(path)
        if not loaded_samples or not loaded_labels:
            raise ValueError(f"Failed to load preprocessed data from {path}. It might be empty or corrupted.")

        # 将加载的数据填充到类的属性中
        self.labels = loaded_labels
        self.samples_time = [s['time'] for s in loaded_samples]
        self.samples_freq = [s['freq'] for s in loaded_samples]
        print(f"Preprocessed data loaded successfully: {len(self.labels)} samples.")


# SEEDDataset 类需要类似修改以使用缓存...

# --- 绘图函数 ---
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1);
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss', markersize=3);
    plt.plot(epochs, history['val_loss'], 'ro-', label='Val Loss', markersize=3);
    plt.title('Loss Curve');
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.grid(True);
    plt.legend();
    plt.xticks(np.arange(0, len(epochs) + 1, step=max(1, len(epochs) // 10)))
    plt.subplot(1, 2, 2);
    plt.plot(epochs, history['train_acc'], 'bo-', label='Train Acc', markersize=3);
    plt.plot(epochs, history['val_acc'], 'ro-', label='Val Acc', markersize=3);
    plt.title('Accuracy Curve');
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.grid(True);
    plt.legend();
    plt.xticks(np.arange(0, len(epochs) + 1, step=max(1, len(epochs) // 10)))
    plt.tight_layout();
    plt.savefig(save_path);
    plt.close()
    print(f"Saved training curves to {save_path}")


# -------------------------
# 3. 主训练函数
# -------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    if args.dataset.upper() == 'DEAP':
        # 传递 preprocessed_data_path
        full_dataset = DEAPDataset(args.data_path, time_window_size=args.time_window, frequence=128,
                                   preprocessed_data_path=args.preprocessed_data_path)
        n_channels, num_classes, time_length = 32, 2, int(args.time_window * 128)
    # SEED 数据集加载逻辑...
    else:
        raise ValueError("Dataset not supported.")

    if len(full_dataset) == 0:
        print("Error: Dataset is empty after loading. Please check data paths and preprocessing logic.")
        return

    val_size = int(len(full_dataset) * 0.2);
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)  # 固定划分种子
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # --- 计算标准化参数 ---
    norm_mean, norm_std = None, None

    # 仅在非恢复训练时计算或检查
    if not (args.resume_from and os.path.exists(args.resume_from)):
        train_indices = train_dataset.indices
        full_dataset.calculate_norm_stats(train_indices)
        norm_mean = full_dataset.mean  # 获取计算出的值 (已是 CPU Tensor)
        norm_std = full_dataset.std  # 获取计算出的值 (已是 CPU Tensor)
    else:
        # 如果恢复训练，尝试从 checkpoint 加载标准化参数
        print(f"Attempting to load normalization stats from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')  # 加载到 CPU
        if 'norm_mean' in checkpoint and 'norm_std' in checkpoint:
            # [FIX 4]: 确保加载的标准化参数始终是 CPU Tensor，因为它们在 DataLoader worker 中使用
            full_dataset.mean = checkpoint['norm_mean'].to('cpu')
            full_dataset.std = checkpoint['norm_std'].to('cpu')
            norm_mean = full_dataset.mean
            norm_std = full_dataset.std
            print("Normalization stats loaded from checkpoint (CPU).")
        else:
            print("Warning: Normalization stats not found in checkpoint, recalculating...")
            train_indices = train_dataset.indices
            full_dataset.calculate_norm_stats(train_indices)
            norm_mean = full_dataset.mean
            norm_std = full_dataset.std

    # [FIX 5]: 移除将 full_dataset.mean/std 移动到 device (CUDA) 的代码。
    # 标准化参数必须保留在 CPU，因为它们在 CPU worker 进程中被访问。

    # [FIX 6]: 修复 DataLoader 的 num_workers 参数
    # 设置 num_workers=0 是解决多进程/CUDA 冲突最可靠的方法
    num_workers_to_use = 0
    print(f"DataLoader num_workers set to {num_workers_to_use} to avoid CUDA/CPU device conflict.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers_to_use,
                              pin_memory=True, drop_last=True)  # drop_last=True 避免BN出错
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers_to_use,
                            pin_memory=True, drop_last=False)

    # --- 模型初始化 ---
    model = GTENet(
        n_channels=n_channels,
        n_freq_bands=5,
        time_length=time_length,
        num_classes=num_classes,
        gat_heads=args.gat_heads,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        d_model=args.d_model,
        # 其他参数可以使用默认值或通过args传入
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=1e-6)

    # --- 加载检查点逻辑 ---
    start_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        # 再次加载以获取其他训练状态，但确保模型状态加载到正确的设备
        checkpoint = torch.load(args.resume_from, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            history = checkpoint.get('history', history)
            # 标准化参数已在前面加载到 full_dataset 对象 (CPU)
            print(f"Resumed from epoch {checkpoint['epoch']}. Starting epoch {start_epoch}.")
        except KeyError as e:
            print(
                f"Error loading checkpoint: Missing key {e}. Checkpoint might be incompatible. Starting from scratch.")
            start_epoch = 0;
            best_val_acc = 0.0;
            epochs_no_improve = 0
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0;
            best_val_acc = 0.0;
            epochs_no_improve = 0
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # --- 训练循环 ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        for batch_idx, (x_time, x_freq, labels) in enumerate(train_pbar):
            # Dataloader 返回的是 CPU 张量，在这里移到 GPU
            x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)
            if random.random() > args.data_augmentation:
                x_time = augment_eeg(x_time,device)
            # 安全检查和数据移动... (保持之前的检查)
            if torch.isnan(x_time).any() or torch.isinf(x_time).any() or \
                    torch.isnan(x_freq).any() or torch.isinf(x_freq).any():
                print(f"Warning: NaN/Inf in input data at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                continue

            optimizer.zero_grad()
            logits = model(x_time, x_freq)
            # 安全检查模型输出... (保持之前的检查)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/Inf in model output at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                continue

            loss = criterion(logits, labels)
            # 安全检查损失值... (保持之前的检查)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN/Inf in loss calculation at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(logits, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                    'acc': f'{correct_preds / total_preds:.4f}' if total_preds > 0 else '0.0000'})

        # 计算 epoch 的平均损失和准确率
        train_acc = correct_preds / total_preds if total_preds > 0 else 0.0
        train_loss = running_loss / total_preds if total_preds > 0 else float('inf')

        # --- 验证阶段 ---
        model.eval()
        val_loss, correct_preds, total_preds = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (x_time, x_freq, labels) in enumerate(val_loader):
                # Dataloader 返回的是 CPU 张量，在这里移到 GPU
                x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)

                # 安全检查... (保持之前的检查)
                if torch.isnan(x_time).any() or torch.isinf(x_time).any() or \
                        torch.isnan(x_freq).any() or torch.isinf(x_freq).any():
                    print(
                        f"Warning: NaN/Inf in validation input data at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                    continue

                logits = model(x_time, x_freq)
                # 安全检查... (保持之前的检查)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(
                        f"Warning: NaN/Inf in validation model output at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                    continue

                loss = criterion(logits, labels)
                # 安全检查... (保持之前的检查)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(
                        f"Warning: NaN/Inf in validation loss calculation at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                    continue

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(logits, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        val_acc = correct_preds / total_preds if total_preds > 0 else 0.0
        val_loss = val_loss / total_preds if total_preds > 0 else float('inf')

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        history['train_loss'].append(train_loss);
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc);
        history['val_acc'].append(val_acc)

        # --- 保存与早停逻辑 ---
        # 仅在 val_loss 有效时才考虑保存和早停
        if val_loss != float('inf'):
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # 生成唯一文件名，避免覆盖
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_path, ext = os.path.splitext(args.model_save_path)
                best_model_path = f"{base_path}_epoch{epoch + 1}_valacc{best_val_acc:.4f}_{timestamp}{ext}"
                print(f"Validation accuracy improved! Saving checkpoint to {best_model_path}")
                # 确保保存的标准化参数是 CPU 张量，方便跨设备加载
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc, 'epochs_no_improve': epochs_no_improve,
                    'history': history,
                    'norm_mean': norm_mean.to('cpu') if norm_mean is not None else None,  # 保存 CPU 版本
                    'norm_std': norm_std.to('cpu') if norm_std is not None else None,  # 保存 CPU 版本
                }, best_model_path)
            else:
                epochs_no_improve += 1
                print(
                    f"Validation accuracy did not improve for {epochs_no_improve} epoch(s). Best was {best_val_acc:.4f}")

            # 学习率调度器步进（基于验证损失）
            scheduler.step(val_loss)

            # 早停检查
            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
                break
        else:
            print(f"Skipping saving and early stopping check due to invalid validation loss in epoch {epoch + 1}.")
            # 即使验证无效，也需要 step scheduler (如果不是 ReduceLROnPlateau)
            # 对于 ReduceLROnPlateau，无效的 loss 不会触发 step

    print(f"Training finished. Best validation accuracy recorded: {best_val_acc:.4f}")

    # 训练结束后绘制曲线图
    if history['train_loss']:  # 确保 history 不为空
        plot_save_path = os.path.splitext(args.model_save_path)[0] + '_final_curves.png'
        plot_curves(history, plot_save_path)


# -------------------------
# 4. 脚本入口
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Resume GTE-Net with Regularization")

    # 数据相关
    parser.add_argument('--dataset', type=str, required=True, choices=['DEAP', 'SEED'], help="Name of the dataset.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the raw dataset directory.")
    parser.add_argument('--preprocessed_data_path', type=str, required=True,
                        help="Path to save/load cached preprocessed data (.npz files per subject).")
    parser.add_argument('--time_window', type=float, default=1.0,
                        help="Length of EEG time window in seconds (e.g., 1.0 for 1s).")  # 改为浮点数，单位秒

    # 模型结构相关
    parser.add_argument('--gat_heads', type=int, default=4, help="Number of GAT heads.")
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of Transformer heads.')
    parser.add_argument('--transformer_layers', type=int, default=3, help="Number of Transformer layers.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for the model.")

    # 训练过程相关
    parser.add_argument('--epochs', type=int, default=150, help="Max number of epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--lr', type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument('--model_save_path', type=str, default='gte_net_best.pth',
                        help="Path template for saving best model.")
    parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint to resume training.")

    # 正则化与优化器
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay (L2 penalty).")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing factor (0 to disable).")

    # 调度器与早停
    parser.add_argument('--scheduler_patience', type=int, default=10, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help="Patience for early stopping (0 to disable).")

    args = parser.parse_args()

    # 确保保存路径目录存在
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train(args)
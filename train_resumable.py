# train_resumable.py
"""
GTE-Net 模型的最终版训练脚本。

新增功能:
-   可以从一个检查点 (checkpoint) 文件恢复训练，加载模型权重、优化器状态和轮次数。
-   使用 --resume_from /path/to/checkpoint.pth 来启用此功能。
-   检查点现在会保存模型、优化器、调度器和epoch信息，以便完美恢复。

保留功能:
-   在 DEAP 或 SEED 数据集上训练。
-   训练结束后自动绘制并保存损失和准确率曲线。
-   包含梯度裁剪、学习率调度器等优化策略。
"""
import os
import argparse
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
from scipy import signal

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR # [新增] 引入学习率调度器
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# 确保导入的是优化版的模型
from model_optimized import GTENet

# -------------------------
# 1. 特征工程 (与之前版本相同)
# -------------------------
def calculate_de(signal_window, dataset_type):
    # (此函数保持不变)
    fs = 200
    if dataset_type == 'SEED': fs = 200
    elif dataset_type == 'DEAP': fs = 128
    
    de_features = []
    bands = [[0.5, 4.], [4., 8.], [8., 13.], [13., 30.], [30., 50.]]
    for low, high in bands:
        b, a = signal.butter(8, [low / (fs / 2), high / (fs / 2)], btype='bandpass')
        filtered = signal.filtfilt(b, a, signal_window)
        var = np.var(filtered)
        if var <= 0: var = 1e-8
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        if np.isnan(de) or np.isinf(de): de = 0.0
        de_features.append(de)
    return np.array(de_features)

# -------------------------
# 2. 数据集类 (与之前版本相同)
# -------------------------
class DEAPDataset(Dataset):
    # (此类保持不变)
    def __init__(self, data_path, n_channels=32, time_window_size=384):
        self.data_path = data_path
        self.n_channels = n_channels
        self.time_window_size = time_window_size
        self.samples = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        print("Loading DEAP dataset...")
        subject_files = [f for f in os.listdir(self.data_path) if f.endswith('.dat')]
        for subject_file in tqdm(subject_files, desc="Processing Subjects"):
            file_path = os.path.join(self.data_path, subject_file)
            with open(file_path, 'rb') as f: subject_data = pickle.load(f, encoding='latin1')
            raw_eeg_data = subject_data['data'][:, :32, :]
            raw_labels = subject_data['labels']
            for i in range(raw_eeg_data.shape[0]):
                trial_eeg = raw_eeg_data[i]
                valence_label = 1 if raw_labels[i][0] >= 5.0 else 0
                for j in range(int(3 * 128 / self.time_window_size), int(trial_eeg.shape[1] / self.time_window_size)):
                    start_point = j * self.time_window_size
                    end_point = start_point + self.time_window_size
                    if end_point > trial_eeg.shape[1]: continue
                    eeg_window = trial_eeg[:, start_point:end_point]
                    x_freq = np.array([calculate_de(ch_data, 'DEAP') for ch_data in eeg_window])
                    self.samples.append({'time': eeg_window, 'freq': x_freq})
                    self.labels.append(valence_label)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        sample, label = self.samples[idx], self.labels[idx]
        return torch.FloatTensor(sample['time']), torch.FloatTensor(sample['freq']), torch.LongTensor([label]).squeeze()

# SEEDDataset 类保持不变...

def plot_curves(history, save_path):
    # (此函数保持不变)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='.')
    plt.plot(history['val_loss'], label='Validation Loss', marker='.')
    plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', marker='.')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='.')
    plt.title('Training & Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"Saved training curves to {save_path}")

# -------------------------
# 3. 主训练函数
# -------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    if args.dataset.upper() == 'DEAP':
        dataset = DEAPDataset(args.data_path, time_window_size=384)
        n_channels, num_classes, time_length = 32, 2, 384
    # SEED 数据集加载逻辑...
    else:
        raise ValueError("Dataset not supported.")

    val_size = int(len(dataset) * 0.2); train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # --- 模型、优化器、调度器初始化 ---
    model = GTENet(n_channels=n_channels, time_length=time_length, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- [新增] 加载检查点逻辑 ---
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0) # 使用.get保证向后兼容
        history = checkpoint.get('history', history) # 恢复历史记录
        
        print(f"Loaded model from epoch {checkpoint['epoch']}. Resuming at epoch {start_epoch}.")
    else:
        if args.resume_from:
            print(f"Checkpoint not found at '{args.resume_from}'. Starting from scratch.")

    # --- [修改] 训练循环从 start_epoch 开始 ---
    for epoch in range(start_epoch, args.epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for x_time, x_freq, labels in train_pbar:
            x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x_time, x_freq)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * x_time.size(0)
            _, predicted = torch.max(logits, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct_preds/total_preds:.4f}'})

        train_acc = correct_preds / total_preds
        train_loss = running_loss / total_preds
        
        # --- 验证阶段 ---
        model.eval()
        val_loss, correct_preds, total_preds = 0.0, 0, 0
        with torch.no_grad():
            for x_time, x_freq, labels in val_loader:
                x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)
                logits = model(x_time, x_freq)
                loss = criterion(logits, labels)
                val_loss += loss.item() * x_time.size(0)
                _, predicted = torch.max(logits, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        val_acc = correct_preds / total_preds
        val_loss = val_loss / total_preds
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        # --- [修改] 保存逻辑，保存完整的检查点 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model found! Saving checkpoint to {args.model_save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history, # 保存历史记录
            }, args.model_save_path)
            
        # 学习率调度器步进
        scheduler.step()

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    plot_save_path = os.path.splitext(args.model_save_path)[0] + '_curves.png'
    plot_curves(history, plot_save_path)

# -------------------------
# 4. 脚本入口
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Resume Training GTE-Net for EEG Emotion Recognition")
    parser.add_argument('--dataset', type=str, required=True, choices=['DEAP', 'SEED'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64) # 增大了默认批次大小
    parser.add_argument('--lr', type=float, default=5e-4) # 稍微增大了默认学习率
    parser.add_argument('--model_save_path', type=str, default='best_model.pth')
    
    # [新增] 用于恢复训练的参数
    parser.add_argument('--resume_from', type=str, default=None, help="Path to a checkpoint file to resume training from.")
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(args.model_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    train(args)
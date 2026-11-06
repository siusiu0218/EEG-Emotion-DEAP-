# train.py
"""
用于在DEAP或SEED数据集上训练GTE-Net模型的训练脚本。

如何运行:
1.  确保你已经将GTE-Net模型代码保存为 `model.py`。
2.  确保DEAP或SEED数据集已经下载并解压。
3.  DEAP 路线可直接使用；SEED 加载逻辑仍需按实际文件结构验证。

示例命令行: # 训练DEAP数据集 python GTE_train.py --dataset DEAP --data_path ./dataset/DEAP/data_preprocessed_python --epochs 700 --model_save_path ./runs/GTEbest/DEAP/GTE_train_model_best.pth --resume_from ./runs/GTEbest/DEAP/GTE_train_model_best_train_acc_0.9216884792248048_val_acc_0.809412981974605.pth

# 训练SEED数据集
python GTE_train.py --dataset SEED --data_path ./dataset/SEED/Preprocessed_EEG --epochs 150 --model_save_path ./runs/GTEbest/SEED/model_best.pth
"""
import os
import argparse
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
import random
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from DatasetProcessSave import save_preprocessed_data,load_preprocessed_data,check_preprocessed_data_exists,calculate_de,calculate_de_batch
# 从 model.py 导入我们设计的模型
from model_optimized import GTENet
import matplotlib.pyplot as plt
# -------------------------
# 1. 特征工程 (伪实现)
# -------------------------

class DEAPDataset(Dataset):
    """用于加载DEAP数据集的PyTorch Dataset类"""
    def __init__(self, data_path, time_window_size , n_channels=32, n_slices=1 , frequence = 128 , preprocessed_data_path = "", subject_ids=None): # n_slices是什么？为什么是8
        self.data_path = data_path # 数据集路径
        self.n_channels = n_channels # 通道数
        self.frequence = frequence
        self.time_window_length = time_window_size*self.frequence # DEAP的时间窗口大小，需要查一查
        print("DEAP数据集时间窗口大小为：",self.time_window_length)
        self.n_slices = n_slices # 这个参数做什么不知道
        self.slice_len = self.time_window_length // n_slices

        self.samples = []
        self.labels = []
        exist = check_preprocessed_data_exists(load_path=preprocessed_data_path,data_path=data_path)
        if not exist:
            save_preprocessed_data(data_path=data_path, save_path=preprocessed_data_path,time_window_length=self.time_window_length)
        self.samples,self.labels = load_preprocessed_data(preprocessed_data_path, subject_ids=subject_ids)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # 转换为torch tensor
        x_time = torch.FloatTensor(sample['time'])
        x_freq = torch.FloatTensor(sample['freq'])
        label = torch.LongTensor([label]).squeeze()
        return x_time, x_freq, label

def augment_eeg(eeg_data,device):
    # 添加随机噪声
    noise = torch.randn_like(eeg_data).to(device) * 0.001
    return eeg_data + noise


# 改进的数据增强方法
def augment_eeg_advanced(eeg_data, device):
    augmented = eeg_data.clone()

    # 1. 高斯噪声
    if random.random() < 0.7:
        noise = torch.randn_like(eeg_data) * 0.001
        augmented += noise

    # 2. 时间扭曲
    if random.random() < 0.5:
        scale_factor = random.uniform(0.9, 1.1)
        new_length = int(eeg_data.shape[-1] * scale_factor)
        resized = F.interpolate(eeg_data, size=new_length, mode='linear', align_corners=False)
        if new_length > eeg_data.shape[-1]:
            augmented = resized[:, :, :eeg_data.shape[-1]]
        else:
            padding = torch.zeros_like(eeg_data)
            padding[:, :, :new_length] = resized
            augmented = padding

    # 3. 通道丢弃 (模拟电极失效)
    if random.random() < 0.3:
        num_channels = eeg_data.shape[1]
        drop_channels = random.sample(range(num_channels), max(1, num_channels // 10))
        augmented[:, drop_channels, :] = 0

    return augmented


class SEEDDataset(Dataset):
    """用于加载SEED数据集的PyTorch Dataset类"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []
        self.labels = []
        
        # SEED标签: +1 (positive), 0 (neutral), -1 (negative)
        self.label_encoder = LabelEncoder()
        self._load_data()
        
    def _load_data(self): # 这个函数需要修改
        print("Loading SEED dataset...")
        # SEED数据集通常是每个被试一个.mat文件
        subject_files = [f for f in os.listdir(self.data_path) if f.endswith('.mat')]
        
        all_labels_for_fit = []
        # 预先收集所有标签以fit LabelEncoder
        for label_file in os.listdir(self.data_path):
            if 'label' in label_file:
                 mat_labels = scipy.io.loadmat(os.path.join(self.data_path, label_file))['label'][0]
                 all_labels_for_fit.extend(mat_labels.tolist())
        self.label_encoder.fit(all_labels_for_fit)

        for subject_file in tqdm(subject_files, desc="Processing Subjects"):
            if 'label' in subject_file: continue
            
            # TODO: 用户需要根据自己的文件命名和结构实现完整的数据加载逻辑
            file_path = os.path.join(self.data_path, subject_file)
            mat_data = scipy.io.loadmat(file_path)
            
            # SEED的.mat文件通常包含了多个试验的数据
            # key通常是类似 `subjectID_experimentNum`
            for key in mat_data.keys(): # 需要修改
                if '__' in key: continue # 跳过mat文件的元信息
                
                # 这是一个试验的数据, 形状 (通道, 时间点)
                trial_data = mat_data[key]
                # SEED的DE特征通常是预计算好的，我们需要找到对应的文件或key
                # 这里我们用伪代码表示，你需要根据实际情况修改
                # de_features = mat_data[key + "_de_LDS"] # -> (通道, 频带)
                
                # --- 伪实现 ---
                # 假设 `trial_data` 包含DE和时域数据，需要切分
                # 例如, 前62列是时域数据，后5列是DE特征
                x_time = trial_data[:, :1024] # 假设时间点为1024
                # x_freq = trial_data[:, -5:]  # 假设DE特征在最后
                
                # 由于我们没有时域信号，这里用随机数代替, 频域用伪实现
                # x_time = np.random.randn(62, 1024)
                x_freq = np.zeros((62, 5))
                for ch in range(x_time.shape[0]):
                    x_freq[ch, :] = calculate_de(x_time[ch, :],dataset_type='SEED')
                # --- 伪实现结束 ---

                self.samples.append({'time': x_time, 'freq': x_freq})

        # 加载标签
        label_path = os.path.join(self.data_path, 'label.mat')
        mat_labels = scipy.io.loadmat(label_path)['label'][0]
        # 每个被试进行了15次试验, 每个文件有3个被试
        # 假设文件按顺序排列
        self.labels = np.repeat(mat_labels, len(subject_files)-1) # 减去label.mat文件
        self.labels = self.label_encoder.transform(self.labels)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        x_time = torch.FloatTensor(sample['time'])
        x_freq = torch.FloatTensor(sample['freq'])
        label = torch.LongTensor([label]).squeeze()
        return x_time, x_freq, label


# -------------------------
# 3. 主训练函数
# -------------------------
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss', markersize=3)
    plt.plot(epochs, history['val_loss'], 'ro-', label='Val Loss', markersize=3)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(0, len(epochs) + 1, step=max(1, len(epochs) // 10)))
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Train Acc', markersize=3)
    plt.plot(epochs, history['val_acc'], 'ro-', label='Val Acc', markersize=3)
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(0, len(epochs) + 1, step=max(1, len(epochs) // 10)))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")

def train(args):
    """主训练和验证流程"""
    if args.resume_from and not os.path.isfile(args.resume_from):
        raise FileNotFoundError(args.resume_from)
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    if args.dataset.upper() == 'DEAP':
        dataset = DEAPDataset(args.data_path, time_window_size=args.time_window, frequence=128,
                              preprocessed_data_path=args.preprocessed_data_path, subject_ids=args.subject_ids)
        n_channels, num_classes, time_length = 32, 2, int(args.time_window * 128)
    elif args.dataset.upper() == 'SEED':
        dataset = SEEDDataset(args.data_path)
        # SEED通道数为62
        n_channels = 62
        num_classes = 3 # positive, neutral, negative
        time_length = 1024 # 伪实现中的长度
    else:
        raise ValueError("Dataset not supported. Choose 'DEAP' or 'SEED'.")

    # 小数据冒烟检查：固定抽取窗口，默认不限制正式数据。
    if args.max_samples is not None:
        if args.max_samples < 10:
            raise ValueError('--max_samples must be at least 10')
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:args.max_samples]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    if len(dataset) < 10:
        raise ValueError('At least 10 samples are required for training and validation')
    # 划分训练集和验证集
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)  # 固定划分种子
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print(f'Window split: total={len(dataset)}, train={train_size}, val={val_size}, seed=42')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("训练加载器的大小为：",len(train_loader))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
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
        ffn_dim=args.ffn_dim,
        # 其他参数可以使用默认值或通过args传入
    ).to(device)
    
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # --- 损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    stopping_patience = 0
    # --- 训练循环 ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        if any(len(values) != start_epoch or not np.isfinite(values).all()
               for values in history.values()):
            raise ValueError('Checkpoint history does not match its epoch')
        if args.epochs <= start_epoch:
            raise ValueError('--epochs must exceed the completed checkpoint epochs')
        best_val_acc = checkpoint['best_val_acc']
        best_index = int(np.argmax(history['val_acc']))
        best_train_acc = checkpoint.get('best_train_acc', history['train_acc'][best_index])
        stopping_patience = checkpoint.get('stopping_patience', 0)
        data_augmentation = args.data_augmentation
        print(f"Resume completed_epochs={start_epoch}, target_epochs={args.epochs}, "
              f"augmentation={data_augmentation}, lr={optimizer.param_groups[0]['lr']}")
    else: # 从头练
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )  # 加入学习率调节器
        best_train_acc = 0.0
        best_val_acc = -1.0  # 首次验证即使准确率为 0 也保存检查点。
        data_augmentation = args.data_augmentation
    print("目前最佳训练与测试精度为：",best_train_acc,best_val_acc)
    os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)
    model_save_path = args.model_save_path
    last_save_path = os.path.splitext(args.model_save_path)[0] + '_last.pth'
    if args.resume_from:
        source_save_path = os.path.splitext(args.model_save_path)[0] + '_resume_source.pth'
        if os.path.realpath(args.resume_from) in map(os.path.realpath, (model_save_path, last_save_path, source_save_path)):
            raise ValueError('Resume output must not overwrite the source checkpoint')
        # Keep the starting state separate: a last checkpoint need not be the best.
        torch.save(checkpoint, source_save_path)
    for epoch in range(start_epoch, args.epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for x_time, x_freq, labels in train_pbar: # 这是一个batch里的全部输入和输出，数量应该是args.batchsize
            # print(x_time.shape, x_freq.shape, labels.shape) # (32,32,128)(32,32,5)(32)
            if random.random() < data_augmentation:
                x_time = augment_eeg_advanced(x_time.to(device),device=device)
            x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)
            optimizer.zero_grad()
            # 前向传播
            logits = model(x_time, x_freq)
            loss = criterion(logits, labels)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            # train_pbar.set_postfix({'loss': f'{running_loss/total_preds*args.batch_size:.4f}', 'acc': f'{correct_preds / total_preds:.4f}'})
        train_acc = correct_preds / total_preds
        train_loss = running_loss / len(train_loader)
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for x_time, x_freq, labels in val_pbar:
                x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)
                
                logits = model(x_time, x_freq)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
                # val_pbar.set_postfix({'loss': f'{val_loss/total_preds*args.batch_size:.4f}', 'acc': f'{correct_preds/total_preds:.4f}'})
        val_acc = correct_preds / total_preds
        val_loss = val_loss / len(val_loader)
        history['train_loss'].append(train_loss);
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc);
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        # --- 保存最佳模型,判断是否早停 ---
        # scheduler.step()
        # if val_acc > best_val_acc or train_acc > best_train_acc:
        if val_acc > best_val_acc:
            stopping_patience = 0
            best_val_acc = val_acc
            best_train_acc = train_acc
            # 保存路径保留下训练精度和测试精度的信息
            model_save_path = args.model_save_path.split('.pth')[0] +f'_train_acc_{train_acc}_val_acc_{val_acc}' +'.pth'
            print(f"New best model found! Saving to {model_save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_train_acc': best_train_acc,
                'stopping_patience': stopping_patience,
                'config': vars(args).copy(),
                'history': history,  # 保存历史记录
            }, model_save_path)
        else:
            stopping_patience += 1
        # Keep the last epoch separately from the best validation checkpoint.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_train_acc': best_train_acc,
            'stopping_patience': stopping_patience,
            'config': vars(args).copy(),
            'history': history,
        }, last_save_path)
        if stopping_patience >= args.early_stopping_patience:
            print(f"在第{epoch + 1}轮早停")
            break
    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    if history['train_loss']:  # 确保 history 不为空
        plot_save_path = model_save_path.split('.pth')[0] + '_final_curves.png'
        plot_curves(history, plot_save_path)

# -------------------------
# 4. 脚本入口
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTE-Net for EEG Emotion Recognition")
    parser.add_argument('--dataset', type=str, required=True, choices=['DEAP', 'SEED'], help="Name of the dataset to use.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the preprocessed dataset directory.")
    parser.add_argument('--epochs', type=int, default=100, help="Total epoch target, including completed epochs when resuming.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--model_save_path', type=str, default='best_model.pth', help="Path to save the best model checkpoint.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 loss on parameters).")
    parser.add_argument('--preprocessed_data_path', type=str, default='./dataset/DEAP/data_time_window_1s', help="切成数据块之后的内容.")
    parser.add_argument('--gat_heads', type=int, default=2, help="GAT头的数量.")
    parser.add_argument('--transformer_heads',type=int,default=4,help='transformer的头数量')
    parser.add_argument('--transformer_layers',type=int,default=1,help="transformer的层数")
    parser.add_argument('--early_stopping_patience',type=int,default=100,help="early stopping patience.")
    parser.add_argument('--d_model',type=int,default=128,help="gte的参数，用于设置分类器.")
    parser.add_argument('--ffn_dim',type=int,default=512,help="gte的参数之一，用于设置transformer")
    parser.add_argument('--data_augmentation',type=float,default=0.5,help="数据增强的概率。")
    parser.add_argument('--resume_from',type=str,default=None,help="继续训练的参数所在的路径。")
    parser.add_argument('--time_window',type=int,default=1,help="模型输入时间窗口大小")
    parser.add_argument('--subject_ids', type=int, nargs='+', default=None, help='DEAP cache subject IDs; omitted loads all subjects.')
    parser.add_argument('--max_samples', type=int, default=None, help='Fixed random window subset for smoke checks only; omitted uses all windows.')
    # parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    args = parser.parse_args()


    train(args)

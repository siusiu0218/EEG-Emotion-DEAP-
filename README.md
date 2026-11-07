# GTE-Net：DEAP 脑电情绪识别复现

基于 [Yuzehua00/GTE-Net](https://github.com/Yuzehua00/GTE-Net) 的学习与研究复现项目，使用 **DEAP 数据集进行 Valence（情绪效价）二分类**。模型融合 EEG 时域信号与五频带差分熵（DE）特征，通过图注意力网络（GAT）和 Transformer 完成分类。

本仓库整理了数据预处理、训练与续训、验证脚本和实验记录，并修复了复现过程中发现的准确率统计、数据增强及运行兼容性问题。模型方案来自上游；本地修改详见[上游变更记录](docs/UPSTREAM_CHANGES.md)。

> **当前结果：最佳窗口级验证 Accuracy 为 75.49%。** 已完成单次训练及续训流程；

[实验结果](#实验结果) · [模型与数据](#模型与数据) · [快速开始](#快速开始) · [文档导航](#文档导航)

## 实验结果

以下实验均为 **DEAP Valence 二分类**：32 名受试者、76,800 个不重叠的 1 秒窗口；以 seed 42 进行窗口级 80/20 随机划分，训练集 61,440 个窗口，验证集 15,360 个窗口。Accuracy 按完整验证集的正确预测数 / 样本数计算，当前没有独立测试集。

| 实验阶段 | 累计训练轮数 | 最佳验证 Accuracy | 验证情况 |
| --- | ---: | ---: | --- |
| 全量冒烟实验 | 1 | 56.76% | 已独立重载并核对完整验证集 |
| 中程训练 | 50 | 64.93%（第 48 轮） | 训练完成，最佳模型未独立重载复核 |
| 续训 | 100 | 69.93%（第 98 轮） | 训练完成，非独立重复实验 |
| 自适应续训 | 378 | **75.49%（第 338 轮）** | 按早停规则结束，最佳模型未独立重载复核 |

50 轮、100 轮和 378 轮结果来自同一训练链的续训分支，**不是多次独立实验的均值**。最终阶段连续 40 轮未达到累计 0.1 个百分点的改善阈值后停止，最佳值为训练日志记录的 75.48828125%。Macro-F1、混淆矩阵及多次运行的均值与标准差尚未报告。

最终续训采用 `batch_size=32`、`weight_decay=1e-4`、`n_slices=1`，从第 100 轮存档恢复；完整超参数和提交命令见[实验日志中的自适应续训记录](docs/EXPERIMENT_LOG.md)。该实验使用基于 `a6facc96b2c159abe14b8d88faf95564d2634122` 未提交工作树派生的冻结副本，并非当前主分支的直接运行结果；代码校验和、命令、环境和终态证据见[实验日志](docs/EXPERIMENT_LOG.md)。

### 如何理解这些结果

- **窗口级划分有信息泄漏风险。** 同一受试者、同一次试验及相邻窗口可能跨训练集和验证集；受试者级独立评估仍待实现。
- **尚未达到上游报告结果。** 项目记录中的上游报告值约为 80.44%；本仓库未复现该数值，固定 700 epochs 配置也未运行。
- **当前 Transformer 序列长度为 1。** 主训练入口使用默认 `n_slices=1`，尚未验证多窗口时序建模能力。
- **随机性未完全固定。** 仅固定了数据划分种子；模型初始化、数据增强等尚未统一设种子，旧检查点也没有 RNG 状态，续训不保证与不中断训练逐步一致。

当前窗口级训练任务已结束。下一阶段拟保留 `subject_id` / `trial_id`，实现受试者级划分、统一随机种子，并补充 Macro-F1、混淆矩阵和重复实验。上述工作尚未实施。

## 模型与数据

### 一个样本是什么

| 输入 / 标签 | 定义 |
| --- | --- |
| 时域输入 | `(32, 128)`：32 个 EEG 通道、128 Hz 采样下的 1 秒信号 |
| 频域输入 | `(32, 5)`：每通道 δ、θ、α、β、γ 五频带 DE 特征 |
| 分类标签 | Valence `< 5` 为 0，`>= 5` 为 1 |

每个受试者包含 40 次试验。代码将每次试验前 3 秒切成三个 1 秒片段并求平均基线，对后 60 秒刺激期按 1 秒切窗、减去基线，再计算特征：

```text
32 名受试者 × 40 次试验 × 60 个窗口 = 76,800 个样本
```

### 模型流程

```mermaid
flowchart LR
    A["时域 EEG · 32 × 128"] --> B[多尺度 1D 卷积]
    C["五频带 DE · 32 × 5"] --> D[全连接投影]
    B --> E[通道特征融合]
    D --> E
    E --> F[两层多头 GAT]
    F --> G[通道注意力读出]
    G --> H["Transformer · 当前序列长度 1"]
    H --> I[时序注意力与二分类]
```

当前标准路线仅使用 DEAP Python 预处理版本。SEED / SEED-IV / SEED-VIG 不在当前复现范围内；`SEEDDataset` 仍包含假设性解析，尚不能视为可复现入口。

## 快速开始

以下命令在 Windows PowerShell 的项目根目录执行。已有检查记录使用 Windows / Python 3.11.9；服务器实验环境见实验日志。

### 1. 准备数据与环境

请自行取得 DEAP Python 预处理数据的使用权限，并将 `s01.dat` 至 `s32.dat` 放到 `dataset/DEAP/data_preprocessed_python/`。仓库不附带数据、预处理缓存或训练权重。

创建 Python 环境：

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果需要 GPU，请先在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择与本机 CUDA 匹配的命令，再安装其余依赖。

### 2. 检查环境、数据和模型

```powershell
.\.venv\Scripts\python.exe validate_setup.py
```

正常输出应包含：

```text
OK: 32 DEAP files
OK: data=(40, 40, 8064), labels=(40, 4)
OK: 60 one-second stimulus windows, baseline excluded
OK: real EEG features time=(32, 128), freq=(32, 5)
OK: model forward output=(1, 2)
```

这个检查只读取 `s01.dat` 并运行一次真实特征计算和前向传播，不会生成全量缓存，也不会开始训练。

### 3. 全量 1 epoch 冒烟实验

```powershell
.\.venv\Scripts\python.exe GTE_train.py `
  --dataset DEAP `
  --data_path .\dataset\DEAP\data_preprocessed_python `
  --preprocessed_data_path .\dataset\DEAP\data_time_window_1s `
  --epochs 1 `
  --batch_size 32 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --time_window 1 `
  --gat_heads 2 `
  --transformer_heads 4 `
  --transformer_layers 1 `
  --d_model 128 `
  --ffn_dim 512 `
  --data_augmentation 0.3 `
  --early_stopping_patience 100 `
  --model_save_path .\runs\smoke\GTE_best.pth
```

第一次运行会依次处理 32 名受试者，并在 `dataset/DEAP/data_time_window_1s/` 生成 `.npz` 缓存。该阶段需要计算 76,800 个窗口的五频带 DE 特征，CPU 上可能耗时较长；之后再次训练会直接复用缓存。

如果显存或内存不足，先把 `--batch_size 32` 改成 `16` 或 `8`。

已运行实验的训练配置、服务器排错与续训命令见[实验日志](docs/EXPERIMENT_LOG.md)。这里的 1 epoch 命令用于检查训练链路，不会复现上表的 378 轮成绩；最终自适应续训使用了独立冻结副本。

## 项目结构

```text
GTE-Net/
├── DatasetProcessSave.py     # 切窗、基线校正、DE 特征与缓存
├── model_optimized.py        # GTE-Net 模型定义
├── GTE_train.py              # 标准训练入口
├── validate_setup.py         # 环境、真实数据与模型前向检查
├── validate_training_runtime.py # scheduler 与数据增强检查
├── validate_smoke_outputs.py # 冒烟实验产物与重载核对
├── jobs/                    # Slurm 作业脚本
├── docs/                    # 实验日志、数据说明与学习文档
├── requirements.txt         # Python 依赖
├── dataset/DEAP/            # 原始数据与缓存（不提交）
└── runs/                    # 检查点与曲线（不提交）
```

`train_regularized.py` 和 `train_resumable.py` 保留用于上游对照，不作为当前标准训练入口。训练产物写入 `--model_save_path` 所在目录，包括检查点和损失 / 准确率曲线；正式实验的配置与结果另记入实验日志。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [最短上手路线](docs/START_HERE_CN.md) | 环境检查与首次运行 |
| [数据目录说明](docs/DATA_LAYOUT_CN.md) | 数据版本、目录和用途 |
| [实验日志](docs/EXPERIMENT_LOG.md) | 实际执行的验证、训练和结果证据 |
| [相对上游的修改](docs/UPSTREAM_CHANGES.md) | 本地修复及其验证边界 |
| [项目状态](CONTEXT.md) | 当前阶段、决策和待完成事项 |
| [学习路线](docs/LEARNING_PLAN.md) | 分步学习与练习安排 |
| [DEAP 操作指南](docs/DEAP脑电预处理操作指南.md) | 数据处理和训练流程详解 |

## 来源与使用说明

- 上游代码：[Yuzehua00/GTE-Net](https://github.com/Yuzehua00/GTE-Net)
- 数据集论文：Koelstra et al., *DEAP: A Database for Emotion Analysis Using Physiological Signals*, 2012. [DOI: 10.1109/T-AFFC.2011.15](https://doi.org/10.1109/T-AFFC.2011.15)
- 图注意力网络：Veličković et al., *Graph Attention Networks*, ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

本项目用于学习与研究复现，不用于临床诊断。使用时请注明上游来源、数据集、实际修改与评估划分。

仓库当前未提供独立的 `LICENSE` 文件；使用与再分发前请确认上游许可及数据集使用条款。数据文件、缓存、检查点和训练曲线不随仓库分发。

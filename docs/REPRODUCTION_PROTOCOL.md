# GTE-Net 复现实验方案

版本：0.1  
整理日期：2026-10-10  
状态：阶段 A 定义已冻结；阶段 B 尚未实现

文档现位于 `docs/`；所有运行命令和数据、缓存、输出路径仍以项目根目录 `D:\workspace\GTE-Net` 为基准。

## 1. 目的

本方案将复现分为两个不能混淆的目标：

- **阶段 A：代码复现。** 按当前上游思路进行窗口级 80/20 随机划分，检查能否跑通并接近上游 README 报告的约 80.44%。
- **阶段 B：可信泛化评估。** 按受试者划分训练、验证和测试，或执行 Leave-One-Subject-Out（LOSO），评估模型面对未见受试者时的表现。

阶段 A 的结果只能说明当前代码流程和上游报告方式的复现程度，不能代替阶段 B。

## 2. 上游与本地版本

- 上游仓库：`https://github.com/Yuzehua00/GTE-Net`
- 上游基线提交：`a6facc96b2c159abe14b8d88faf95564d2634122`
- 本地改动：见 `docs/UPSTREAM_CHANGES.md`
- 每次正式实验都必须在 `docs/EXPERIMENT_LOG.md` 记录当前 `git rev-parse HEAD` 和 `git status --short`。

工作树未提交时，必须明确写“dirty working tree”，不能只记录 HEAD。

## 3. 数据定义

### 3.1 当前唯一正式输入

```text
dataset/DEAP/data_preprocessed_python/s01.dat ... s32.dat
```

预期条件：

- 32 个受试者文件；
- 单个文件 `data` 形状为 `(40, 40, 8064)`；
- 单个文件 `labels` 形状为 `(40, 4)`；
- 前 32 个信号通道作为 EEG；
- 采样率 128 Hz；
- 每次试验包含 3 秒基线和 60 秒刺激期。

当前没有保存文件级校验和。正式发表或跨机器迁移前，应生成数据清单与 SHA-256，并将清单提交到 Git，但不得提交数据本体。

### 3.2 当前不使用的数据

- DEAP Matlab 预处理版；
- DEAP 原始 BDF；
- DEAP 面部视频和辅助元数据；
- SEED、SEED-IV、SEED-VIG。

这些数据的保存位置和用途见 `docs/DATA_LAYOUT_CN.md`。

## 4. 任务与标签

- 任务：DEAP Valence 二分类；
- 原始标签字段：`labels[:, 0]`；
- 阈值：Valence `< 5.0` 记为 0，Valence `>= 5.0` 记为 1；
- 每次试验的标签复制到该试验产生的全部窗口。

不得把 Valence 二分类结果描述为完整的“情绪识别性能”，也不得与 Arousal、四分类或 SEED 三分类结果直接比较。

## 5. 预处理定义

以 `--time_window 1` 为标准设置：

1. 读取每次试验前 32 个 EEG 通道；
2. 前 3 秒，即前 384 个采样点，作为基线；
3. 将 3 秒基线切成三个 128 点片段，在对应时间位置上求平均，得到 `(32, 128)` 的平均基线；
4. 从采样点 384 开始，只处理后 60 秒刺激信号；
5. 使用互不重叠的 128 点窗口，末个窗口起点为 7936；
6. 每次试验得到 60 个窗口；
7. 总窗口数应为 `32 × 40 × 60 = 76,800`；
8. 每个窗口减去平均基线；
9. 保存时域输入 `(32, 128)` 和频域输入 `(32, 5)`。

缓存位置：

```text
dataset/DEAP/data_time_window_1s/subject_01.npz ... subject_32.npz
```

若改变窗口长度、步长、基线方法或频带定义，必须使用新的缓存目录，不能复用旧缓存。

## 6. DE 特征定义

每个 EEG 通道分别计算五频带差分熵：

| 频带 | 范围 |
| --- | --- |
| δ | 0.5–4 Hz |
| θ | 4–8 Hz |
| α | 8–13 Hz |
| β | 13–30 Hz |
| γ | 30–50 Hz |

当前全量预处理由 `calculate_de` 执行：

- 8 阶 Butterworth 带通滤波；
- `scipy.signal.filtfilt` 双向滤波；
- 方差使用 NumPy 默认总体方差；
- `DE = 0.5 × log(2πe × variance)`；
- 非正方差替换为 `1e-8`；
- NaN 或无穷值替换为 0。

`calculate_de_fast` 和 `calculate_de_batch` 使用 `lfilter`，与 `calculate_de` 并非完全相同的滤波过程。若切换实现，必须作为单独实验记录，不能默认认为特征完全等价。

## 7. 阶段 A：窗口级代码复现

### 7.1 数据划分

- 全部 76,800 个窗口合并后进行 `random_split`；
- 训练集 80%，验证集 20%；
- 划分 generator seed 为 42；
- 当前没有独立测试集；
- 当前没有统一固定 Python、NumPy、模型初始化和数据增强随机种子。

因此，阶段 A 存在同一受试者、同一次试验和相邻窗口跨训练/验证集合的可能性。

### 7.2 冻结训练配置

| 参数 | 值 |
| --- | ---: |
| `epochs` | 冒烟 1；中程 50；上游长训练 700 |
| `batch_size` | 32 |
| `lr` | `1e-4` |
| `weight_decay` | `1e-4` |
| `time_window` | 1 秒 |
| `gat_heads` | 2 |
| `transformer_heads` | 4 |
| `transformer_layers` | 1 |
| `d_model` | 128 |
| `ffn_dim` | 512 |
| `data_augmentation` | 0.3 |
| `early_stopping_patience` | 100 |
| 损失 | CrossEntropyLoss，label smoothing 0.1 |
| 优化器 | AdamW |
| 学习率调度 | ReduceLROnPlateau，监控验证损失 |

当前主训练入口没有显式传入 `n_slices`，模型使用默认 `n_slices=1`。Transformer 因而处理长度为 1 的序列。

### 7.3 冻结运行命令

先检查：

```powershell
.\.venv\Scripts\python.exe validate_setup.py
```

再运行 1 epoch：

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

后续只改变 `--epochs` 和输出目录：

- 中程检查：`--epochs 50`，输出到 `runs/deap_window_split_50/`；
- 上游长训练：`--epochs 700`，输出到 `runs/deap_window_split_700/`。

学院服务器 50 epochs 作业入口：`sbatch --nodelist=c002 jobs/gte_deap_50_cpu.slurm`。
从头训练，未传 `--resume_from`，仅改变 epochs 与输出目录；完整训练参数在该脚本内。
每次输出隔离为 `runs/deap_window_split_50/<SLURM_JOB_ID>/`，保留作业脚本副本、
训练代码校验和及带日期的依赖快照。8 CPU、10G、CPU 分区、24 小时上限。

### 7.4 阶段 A 验收门槛

2026-10-15 补充：全量冒烟前可用 `--subject_ids 1 --max_samples 320` 做小数据检查。
读取 subject_01 缓存，用 seed 42 随机抽取 320 个窗口，再按窗口级 80/20 划分为
256/64；其余参数保持冻结设置。省略两个选项即恢复全部数据。
小数据结果仅检查训练链路，不能作为性能结论。服务器顺序作业脚本
`jobs/gte_repair_smoke_cpu.slurm` 执行回归检查、真实前向、小数据 1 epoch，
并重载检查点核对全部 64 个验证预测的 Accuracy；全部通过后才启动全量 1 epoch。
完整命令：`sbatch jobs/gte_repair_smoke_cpu.slurm`。输出目录：
`runs/repair/<job_id>/small/`、`runs/repair/<job_id>/full/`。
全量结束后执行 `python validate_smoke_outputs.py --run_dir runs/repair/<job_id>/full --full`，
重载检查点，对全部 15,360 个验证窗口重新预测，并核对保存的 Accuracy。

- Gate A0：`validate_setup.py` 全部检查通过；
- Gate A1：32 个缓存文件完整生成，总样本数为 76,800；
- Gate A2：1 epoch 训练、验证、检查点和曲线保存均正常完成；
- Gate A3：50 epochs 损失和准确率趋势合理，无 NaN/Inf；
- Gate A4：运行冻结的长训练设置，并把真实结果与上游约 80.44% 并列报告。

“接近 80.44%”不预先规定容差。必须先完成重复运行和随机性控制，再解释差异来源。

## 8. 阶段 B：受试者级评估

状态：**尚未实现，不得填入结果。**

实现前至少需要：

- 每个窗口保留 `subject_id` 和 `trial_id`；
- 确保一个受试者不会跨训练、验证或测试集合；
- 冻结一种主分析方案，例如 LOSO；
- 只使用训练集计算任何归一化参数；
- 统一设置 Python、NumPy、PyTorch CPU/CUDA 随机种子；
- 报告 Accuracy、Macro-F1、混淆矩阵和各受试者结果；
- 汇总均值、标准差或置信区间。

阶段 B 的协议细节必须在写代码前追加到本文件并冻结版本号。

## 9. 实验记录与结果表述

每次运行都在 `docs/EXPERIMENT_LOG.md` 新增一条记录，至少包括：

- 实验 ID 和日期；
- 目的；
- Git HEAD 和工作树状态；
- Python、PyTorch、CUDA 和设备；
- 数据与缓存目录；
- 完整命令；
- 开始/结束状态与耗时；
- 样本或受试者划分；
- 最佳 epoch 和指标；
- 输出文件；
- 异常、偏差和结论边界。

只有实际运行产生的数值才能进入结果栏。上游值必须标注“上游报告”，计划值必须标注“待运行”。



## 10. 2026-10-16：从最佳第 48 轮续训至总计 100 轮（冻结配置）

用户明确选择接着已有模型训练，总轮数为 100，不是额外再跑 100 轮。
源检查点为 `runs/deap_window_split_50/1887646/GTE_best_train_acc_0.6437174479166666_val_acc_0.6492838541666667.pth`。
该文件保存的是第 48 轮（epoch=47），因此本次运行第 49–100 轮，共 52 个新 epoch。
前次第 49、50 轮的训练轨迹不接入本次分支；不能称为从第 50 轮无缝续训。

- 数据与任务：32 受试者，76,800 个一秒 DEAP 窗口，Valence <5/≥5 二分类。
- 划分：保持完整数据原顺序和窗口级 random_split seed 42，训练 61,440、验证 15,360。
- 全部模型、优化、增强参数沿用 50 轮配置（见第 7 节）；`--epochs 100` 表示含历史的总轮数。
- 恢复 model、AdamW、ReduceLROnPlateau、history 和历史最佳 Accuracy；增强概率保持 0.3，不采用旧续训分支硬编码的 0.9。
- 源文件未存随机数状态，且全局随机性未统一固定；本次不承诺与未中断训练逐步一致，也不是独立重复实验。
- 修复后每轮另存 `GTE_best_last.pth`，用于保留最新状态；`*_train_acc_*_val_acc_*.pth` 仍为新最佳，`GTE_best_resume_source.pth` 是本次起点副本。
- 所有服务器 Python 计算通过 sbatch 分配资源后，由 srun 在计算节点执行。登录节点仅进行文件传输、脚本语法检查、校验和与调度查询/提交。
- 资源沿用 8 CPU、10G、cpu 分区、24 小时上限。脚本 `jobs/gte_deap_resume100_cpu.slurm`；原申请未指定节点，c001 依赖读取等待后取消，改用 --nodelist=c002，由调度器分配原先跑通的节点。
- 提交前冻结代码到 `runs/resume100_code_20261016_v1/`，运行使用该目录，不覆盖服务器先前训练入口。
- 完整提交命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 runs/resume100_code_20261016_v1/gte_deap_resume100_cpu.slurm /share/home/ycg_luanjingjie/GTE-Net/runs/resume100_code_20261016_v1`。
- 作业内先运行续训回归、scheduler/增强检查，以及 32 个真实窗口（26 训练/6 验证）的 1→2 轮续训检查；任何一步失败即退出，不进入正式续训。
- 正式输出隔离于 `runs/deap_window_split_resume100/<job_id>/`；保留脚本、代码与源检查点校验和、旧依赖快照标记。

本节为实验定义，不代表已经提交或完成；实际调度、检查和训练状态只见实验日志。


## 11. 2026-10-19：从100轮状态续训，按实质改善早停

用户在侧聊明确授权实施并提交集群。使用独立冻结副本 `runs/adaptive_code_20261019_v1/`，根目录研究代码及旧冻结代码保持原样，避免干扰其他工作。

- DEAP Valence 二分类，32人、76,800个1秒窗口；原窗口随机80/20（seed42），61,440训练/15,360验证。数据、标签、模型、增强与指标定义保持协议第10节配置；没有增加标准化或改变窗口。
- 起点是1887720的 `GTE_best_last.pth`（已完成第100轮），从第101轮继续；模型、AdamW、scheduler与100轮history恢复。学习率继承为5e-5，而不是重新设回1e-4。
- 总轮数上限2000；`--early_stopping_patience 40 --early_stopping_min_delta 0.001`（准确率小数单位，0.1个百分点）。
- 新规则初始参考值是历史最好验证Accuracy 0.6993489583333333；起点没有新规则状态，因此从本次续训开始计等待轮次，不把旧规则计数混入。
- 每轮Accuracy相对上次实质改善参考值累计增加至少0.001才更新参考并重置等待；不足阈值的小进步可继续累计。连续40轮未达到阈值即停。
- 保存最佳权重仍按任何严格更高的Accuracy，不受min_delta限制。每轮独立保存最新状态（含早停参考/计数/规则）；采用临时文件+原子替换，保留上次完整存档。
- 原第98轮历史最好模型另复制到新输出目录 inherited_best_epoch98.pth，防止续训没有新最佳时丢失历史最好权重。
- 沿用验证loss的ReduceLROnPlateau（patience10、factor0.5、min_lr1e-6）；早停监控Accuracy，两者依据不同，不直接根据短期斜率停止。
- 仍以sbatch申请cpu分区c002、8 CPU、10G、24小时；所有Python通过srun在分配节点执行。`--signal=USR1@600`提前约10分钟发信号，Python在当前epoch结束、保存最新状态后退出；若单轮超过剩余时间，仍可能被调度器到时终止，之前完整存档保留。
- 本次只提交一份作业，不自动连锁提交后续；2000轮上限与24小时上限独立，若先到时间上限须按存档和日志决定下一次续训。
- 完整提交命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 runs/adaptive_code_20261019_v1/gte_adaptive_cpu.slurm /share/home/ycg_luanjingjie/GTE-Net/runs/adaptive_code_20261019_v1`。
- 前置门槛：累计改善/耐心/状态恢复测试，真实train()小模型早停/续训及Linux时限信号检查，scheduler/增强回归，32真实DEAP窗口1→2轮续训；任何失败均不进入正式训练。
- 输出 `runs/deap_window_split_adaptive/<job_id>/`；退出原因区分 early_stopping / max_epochs / walltime_signal，不把一次作业正常退出写成完成2000轮。

这是同一验证集上的续训分支，不是独立重复或未见受试者测试。旧存档无RNG状态，不能承诺与不中断训练逐步一致。实际状态见实验日志，不预填新成绩。

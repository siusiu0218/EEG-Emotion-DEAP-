# GTE-Net 复现项目

2026-10-15 服务器排错路线：先执行 `validate_training_runtime.py`，再以
`--subject_ids 1 --max_samples 320` 运行 1 epoch（256 训练 / 64 验证窗口）。
`validate_smoke_outputs.py --run_dir <small输出目录>` 重载检查点、核对验证准确率和产物。
全量 1 epoch 结束后，可加 `--full` 对全部 15,360 个验证窗口重新预测核对。
省略两个数据限制选项即使用全部数据。学院服务器可执行
`sbatch jobs/gte_repair_smoke_cpu.slurm`，小数据检查通过后自动启动全量 1 epoch；
配置与限制见 `docs/REPRODUCTION_PROTOCOL.md`，实际状态见 `docs/EXPERIMENT_LOG.md`。

本仓库用于学习和复现 [Yuzehua00/GTE-Net](https://github.com/Yuzehua00/GTE-Net)。GTE-Net 是一个面向脑电（EEG）情绪识别的 PyTorch 实验模型，将原始时域信号、五频带差分熵（Differential Entropy, DE）特征、图注意力网络（GAT）和 Transformer 组合起来完成情绪分类。

本地第一阶段只复现 **DEAP 数据集上的 Valence 二分类**。DEAP 数据已经准备好，环境、真实数据读取、特征计算和模型前向传播均已检查通过；学院服务器已完成全量1 epoch、50轮、100轮和总计378轮的窗口级训练，实际结果见下文与实验日志。

2026-10-22状态同步：本次窗口级训练任务已完成。作业1887873于2026-10-20 12:12:39正常结束（COMPLETED、0:0、耗时17:11:57），总计378轮，第338轮最佳验证Accuracy为75.48828125%，末轮日志75.28%；按40轮/0.1个百分点规则早停。最佳模型、last存档及曲线已在服务器生成。用户决定不做最佳模型独立重载复核，该项未执行。结果是32人、76,800窗口、61,440/15,360窗口级划分下的单次续训成绩，不是跨受试者结果；整个复现项目尚未完成。

2026-10-15 学院服务器进展：320 窗口小数据检查通过，随后全部 32 名受试者、
76,800 窗口的 1 epoch 训练验证完成（作业 1887643，退出码 0）。窗口级 80/20
验证 Accuracy 为 56.7578125%，单次结果；不代表受试者级泛化，也不是上游 80.44% 的复现。
精确配置、代码状态及产物见 `docs/EXPERIMENT_LOG.md`。

2026-10-16 核对：50 epochs 作业 1887646 已于 10 月 15 日 21:22 成功结束，耗时
3:03:14；第 48 epoch 的最佳窗口级验证 Accuracy 为 64.92838541666667%，
最后第 50 epoch 为 64.34%（日志四舍五入）。全部 32 名受试者、76,800 个窗口，
窗口级 80/20 划分，单次结果；模型与曲线位于服务器
`runs/deap_window_split_50/1887646/`。完整命令为
`sbatch --nodelist=c002 jobs/gte_deap_50_cpu.slurm`，从头训练，沿用冻结参数。
50 epochs 最佳模型尚未独立重载核对；700 epochs、重复运行及受试者级评估未完成，
不能据此称整个项目复现完成。完整证据与配置见实验日志。

> 这是一份研究代码复现，不是开箱即用的软件，也不应用于医疗诊断。原作者报告的约 80.4% 是窗口级随机划分结果，不能直接解释为对新受试者的泛化能力。

2026-10-16 新安排：用户选择从 50 轮实验的第 48 轮最佳模型继续到总计 100 轮。
调度入口为 `jobs/gte_deap_resume100_cpu.slurm`，先在计算节点检查续训，再运行第 49–100 轮。
完整提交命令、冻结代码目录与参数见 `docs/REPRODUCTION_PROTOCOL.md` 第 10 节；实际状态见实验日志。
作业1887720已完成100轮续训，第98轮最佳验证Accuracy为69.9349%；随后1887873继续至总计378轮并早停。这些是同一训练链的续训分支，不是独立重复实验。

## 目录

- [当前复现范围](#当前复现范围)
- [模型与数据](#模型与数据)
- [项目结构](#项目结构)
- [指导性文件](#指导性文件)
- [快速开始](#快速开始)
- [训练输出](#训练输出)
- [复现结果记录](#复现结果记录)
- [已知限制](#已知限制)
- [后续实验路线](#后续实验路线)
- [来源与引用](#来源与引用)

## 当前复现范围

| 项目 | 当前状态 |
| --- | --- |
| DEAP Python 预处理数据 | 已就位：`s01.dat` 至 `s32.dat` |
| 数据完整性和形状检查 | 已通过 |
| 真实 EEG 的基线校正和 DE 特征计算 | 已通过 |
| GTE-Net 前向传播 | 已通过 |
| 1 epoch 全链路训练 | 学院服务器已通过，独立重载核对通过 |
| 50 epochs 窗口级实验 | 已完成；单次最佳验证 Accuracy 64.93%，最佳模型待独立重载核对 |
| 100轮及自适应续训 | 已完成；总计378轮早停，第338轮最佳窗口级验证Accuracy 75.49%；未独立重载复核，用户决定不做本次复核 |
| 作者窗口级结果复现 | 尚未完成；700 epochs 未运行，上游约 80.44% 未复现 |
| 受试者级独立评估 | 待实现 |
| SEED / SEED-IV / SEED-VIG | 不在第一阶段范围内 |

已验证的本地环境为 Windows、Python 3.11.9。检查时 PyTorch 使用 CPU；如果本机有 NVIDIA GPU，请按自己的 CUDA 版本安装 PyTorch。

## 模型与数据

### 模型输入与流程

当前 DEAP 配置中，一个样本由以下内容组成：

- 时域输入：`(32, 128)`，即 32 个 EEG 通道的 1 秒信号；
- 频域输入：`(32, 5)`，即每个通道在 δ、θ、α、β、γ 五个频带上的 DE 特征；
- 标签：Valence `< 5` 为 0，Valence `>= 5` 为 1。

主要计算流程为：

```text
1 秒 EEG 窗口 ──> 多尺度 1D 卷积 ──┐
                                    ├─> 通道节点特征
五频带 DE 特征 ──> 全连接投影 ──────┘
                                    │
                                    v
                         两层多头 GAT + LayerNorm
                                    │
                                    v
                              通道注意力读出
                                    │
                                    v
                         Transformer + 时序注意力
                                    │
                                    v
                              Valence 二分类
```

需要注意：当前训练脚本没有显式传入 `n_slices`，模型使用默认值 `n_slices=1`。因此 Transformer 实际接收的序列长度为 1，现阶段不应把它解释为已经学习了多个连续窗口之间的动态关系。

### DEAP 数据

本项目使用 DEAP 官方 Python 预处理版本。每个受试者文件的实测结构为：

- `data`: `(40, 40, 8064)`；
- `labels`: `(40, 4)`；
- 40 次试验，前 32 个通道为 EEG；
- 每次试验包含 3 秒基线和 60 秒刺激期，采样率为 128 Hz；
- 四个评分依次为 Valence、Arousal、Dominance、Liking。

代码跳过前 3 秒基线，将后 60 秒刺激期切成不重叠的 1 秒窗口。32 名受试者共生成：

```text
32 人 × 40 次试验 × 60 个窗口 = 76,800 个窗口样本
```

数据目录应保持为：

```text
dataset/
└── DEAP/
    └── data_preprocessed_python/
        ├── s01.dat
        ├── s02.dat
        ├── ...
        └── s32.dat
```

仓库中的 `datas/archives/` 保存原始下载包，但不会提交到 Git。Matlab 预处理版、原始 BDF、面部视频和 SEED 系列数据都不用于当前 DEAP 快速复现。更完整的说明见 [`docs/DATA_LAYOUT_CN.md`](docs/DATA_LAYOUT_CN.md)。

## 项目结构

```text
GTE-Net/
├── README.md                 # 本复现说明
├── CONTEXT.md                # 项目目标、决策、状态和下一步
├── AGENTS.md                 # AI/协作者工作边界
├── docs/                     # 学习与复现文档、个人笔记
│   ├── LEARNING_PLAN.md      # 学习路线与验收方式
│   ├── LEARNING_PROGRESS.md  # 当前学习停点与下一步
│   ├── MISSION.md            # 独立复现学习目标
│   ├── NOTES.md              # 教学偏好
│   ├── RESOURCES.md          # 学习资源
│   ├── START_HERE_CN.md      # 最短上手路线
│   ├── DATA_LAYOUT_CN.md     # 数据目录和压缩包说明
│   ├── REPRODUCTION_PROTOCOL.md # 冻结实验定义
│   ├── EXPERIMENT_LOG.md     # 实际验证与训练记录
│   ├── UPSTREAM_CHANGES.md   # 相对上游的本地修改
│   ├── DEAP脑电预处理操作指南.md
│   └── 小小学习笔记——仅供自己观看.md
├── lessons/                 # HTML 小课
├── reference/               # 术语与速查材料
├── assets/                  # 课程共用样式
├── requirements.txt          # Python 依赖
├── validate_setup.py         # 只读的环境、数据和模型检查
├── DatasetProcessSave.py     # DEAP 切窗、基线校正、DE 特征和缓存
├── model_optimized.py        # GTE-Net 模型定义
├── GTE_train.py              # 当前主训练入口
├── train_regularized.py      # 上游实验性正则化版本
├── train_resumable.py        # 上游实验性断点续训版本
├── dataset/                  # 已解压数据，Git 忽略
├── datas/                    # 下载归档，Git 忽略
└── runs/                     # 检查点和训练曲线，Git 忽略
```

第一阶段只使用 `GTE_train.py`。另外两个训练脚本保留用于对照，不作为当前复现的标准入口。

## 指导性文件

根目录保留总览、协作规则和项目状态；其余 Markdown 文档集中在 `docs/`。文档中的命令和反引号中的项目路径默认以项目根目录 `D:\workspace\GTE-Net` 为基准，点击链接则按各文档的相对位置解析。

- [学习计划](docs/LEARNING_PLAN.md)：完整学习路线与出题、验收规则；
- [学习进度](docs/LEARNING_PROGRESS.md)：继续做题时先读，确认当前停点和下一步；
- [DEAP 操作指南](docs/DEAP脑电预处理操作指南.md)：数据处理与训练流程的详细解释；

- [`CONTEXT.md`](CONTEXT.md)：用一页说明为什么做、做到哪里、当前决策和下一步；
- [`docs/START_HERE_CN.md`](docs/START_HERE_CN.md)：第一次进入仓库时使用的最短路线；
- [`docs/DATA_LAYOUT_CN.md`](docs/DATA_LAYOUT_CN.md)：说明数据放在哪里、哪些数据当前使用；
- [`AGENTS.md`](AGENTS.md)：约束 AI 和协作者的修改、验证和结果表述；
- [`docs/REPRODUCTION_PROTOCOL.md`](docs/REPRODUCTION_PROTOCOL.md)：冻结数据、标签、预处理、划分、参数和验收标准；
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)：只记录真实运行过的检查和训练；
- [`docs/UPSTREAM_CHANGES.md`](docs/UPSTREAM_CHANGES.md)：区分原作者实现与本地复现修改。

## 快速开始

以下命令在项目根目录执行。

### 1. 创建环境

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

### 3. 跑 1 epoch 冒烟实验

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

### 4. 进行完整复现

建议先跑 50 epochs，确认损失下降、准确率变化和检查点保存都正常，再尝试作者配置中的 700 epochs：

```powershell
.\.venv\Scripts\python.exe GTE_train.py `
  --dataset DEAP `
  --data_path .\dataset\DEAP\data_preprocessed_python `
  --preprocessed_data_path .\dataset\DEAP\data_time_window_1s `
  --epochs 50 `
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
  --model_save_path .\runs\deap_window_split\GTE_best.pth
```

将 `--epochs 50` 改为 `--epochs 700` 即可运行作者 README 中的长训练设置。不要一开始就跑 700 epochs；先用 1 epoch 和 50 epochs 分别验证链路与收敛趋势。

## 训练输出

当验证准确率提高时，脚本会在 `--model_save_path` 所在目录保存检查点，文件名中包含当时的训练和验证准确率，例如：

```text
GTE_best_train_acc_0.90_val_acc_0.79.pth
```

训练结束后还会保存损失和准确率曲线 PNG。缓存、检查点和曲线都属于可再生的大文件或运行产物，默认不会提交到 Git。

建议为每次正式实验另外保存：

- 完整命令和超参数；
- Python、PyTorch、CUDA 和显卡版本；
- 随机种子；
- 最佳 epoch、Accuracy、Macro-F1 和混淆矩阵；
- 数据划分方式（窗口级或受试者级）。

## 复现结果记录

原作者 README 报告 DEAP Valence 二分类的最佳验证准确率约为 **80.44%**。这个数字是上游报告值，不是本仓库当前已经重新跑出的结果。

| 实验 | 数据划分 | Epochs | 最佳验证 Accuracy | Macro-F1 | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| 环境与前向检查 | 单个真实窗口 | 0 | — | — | 已通过 |
| 全量冒烟实验，1887643 | 窗口级 80/20 | 1 | 56.76% | 未计算 | 通过，独立重载核对通过 |
| 中程实验，1887646 | 窗口级 80/20 | 50 | 64.93%（第 48 轮） | 未计算 | 训练完成，最佳模型待独立重载核对 |
| 续训，1887720 | 窗口级 80/20 | 总计100 | 69.93%（第98轮） | 未计算 | 训练完成；非独立重复 |
| 自适应续训，1887873 | 窗口级 80/20 | 总计378 | 75.49%（第338轮） | 未计算 | 训练完成，按规则早停；未独立重载复核 |
| 上游长训练设置 | 窗口级 80/20 | 700 | 待填写 | 待填写 | 待运行 |
| 可信泛化评估 | 受试者级 | 待定 | 待填写 | 待填写 | 待实现 |

完成实验后，应把真实运行命令、环境和结果补入本表，避免把上游结果误写成本地结果。

## 已知限制

1. **当前划分存在信息泄漏风险。** `GTE_train.py` 使用固定种子 42 对 76,800 个窗口做 80/20 `random_split`。同一受试者、同一次试验甚至相邻窗口可能同时进入训练集和验证集。
2. **约 80.4% 不是跨受试者性能。** 它适合作为“代码能否接近上游窗口级结果”的复现指标，不能代表模型对新受试者的实际泛化。
3. **Transformer 当前序列长度为 1。** 主训练入口没有设置大于 1 的 `n_slices`，因此时序建模能力需要进一步核验和改造。
4. **随机性没有完全固定。** 数据划分固定了 PyTorch generator，但 Python、NumPy、模型初始化和数据增强的随机种子没有统一设置，重复运行可能得到不同结果。
5. **SEED 路线尚未就绪。** `SEEDDataset` 中仍有假设性解析和待修改逻辑，不应把“数据已经下载”当作 SEED 已可复现。
6. **续训已完成，但不保证与不中断训练逐步一致。** 小数据续训检查、正式100轮及总计378轮早停训练已完成。旧checkpoint没有随机数状态，且全局随机性未统一固定；本次最佳模型未独立重载复核，用户决定不开展该项。
7. **当前只有 Accuracy。** 主训练脚本尚未输出 Macro-F1、混淆矩阵、灵敏度或特异度。

## 后续实验路线

建议按以下顺序推进：

截至2026-10-22，第1、2步已通过，第3步及后续100轮/总计378轮续训已完成；本次窗口级训练任务收尾。用户决定不做最佳模型独立重载复核。第4步的固定700轮配置及上游约80.44%尚未复现，第5至8步仍未实施；后续如继续研究，先确定受试者级评估方案。以下是研究路线，不代表已执行或已授权启动新实验。

1. 运行 `validate_setup.py`；
2. 跑通 1 epoch，确认缓存、训练、验证和保存链路；
3. 跑 50 epochs，检查是否正常收敛；
4. 运行作者窗口级配置，记录与约 80.44% 的差距；
5. 给每个窗口保留 `subject_id` 和 `trial_id`；
6. 改为按受试者切分，或进行 Leave-One-Subject-Out；
7. 统一随机种子，并报告多次运行的均值和标准差；
8. 增加 Macro-F1、混淆矩阵和各受试者结果；
9. 最后再决定是否修复并复现标准 SEED 三分类。

只有第 4 步是在复现上游报告方式；第 5 至 8 步是在把“代码跑通”升级为更可信的研究实验。

## 来源与引用

- 原始代码仓库：[Yuzehua00/GTE-Net](https://github.com/Yuzehua00/GTE-Net)
- DEAP 数据集：Koelstra et al., *DEAP: A Database for Emotion Analysis Using Physiological Signals*, IEEE Transactions on Affective Computing, 2012. [DOI: 10.1109/T-AFFC.2011.15](https://doi.org/10.1109/T-AFFC.2011.15)
- SEED 数据集主页：[SJTU BCMI SEED](https://bcmi.sjtu.edu.cn/home/seed/)
- Graph Attention Networks：Veličković et al., ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

如果使用本仓库开展研究，请同时引用原始 GTE-Net 仓库和所用数据集论文，并在报告中明确说明本地修改、数据划分和实际复现结果。

## 许可证说明

当前仓库没有独立的 `LICENSE` 文件。原作者 README 将项目限定为研究与教育用途，但这不等同于标准开源许可证授权。公开发布、再分发或商业使用前，请向原作者确认代码许可，并遵守 DEAP、SEED 等数据集各自的使用条款；不要把数据文件提交到公开仓库。


2026-10-19 自适应续训尝试：从1887720的第100轮last存档继续，上限2000轮，连续40轮累计改善不足0.1个百分点则早停。使用独立冻结副本，不修改根目录训练入口；通过Slurm在计算节点执行，单次24小时并请求提前保存退出。完整配置见 docs/REPRODUCTION_PROTOCOL.md 第11节，实际作业状态见 docs/EXPERIMENT_LOG.md。

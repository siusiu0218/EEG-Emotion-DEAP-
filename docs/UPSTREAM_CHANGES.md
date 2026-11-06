# 相对上游 GTE-Net 的本地修改

对照日期：2026-10-10  
上游仓库：`https://github.com/Yuzehua00/GTE-Net`  
对照提交：`a6facc96b2c159abe14b8d88faf95564d2634122`

## 说明

本文件用于区分：

- 上游作者原始实现；
- 为让本地 DEAP 复现路径可检查、可运行而进行的修改；
- 尚未验证或计划中的改进。

当前修改仍在本地工作树中，尚未形成新的本地提交。以下内容根据当前 `git diff` 和新增文件整理；后续提交代码时应更新提交号与状态。

## 已修改的上游文件

### `.gitignore`

新增忽略项：

- `datas/`：原始下载归档；
- `.venv/`：本地 Python 环境；
- `__pycache__/`：Python 缓存。

目的：避免数据、环境和运行缓存误入版本控制。

### `DatasetProcessSave.py`

本地修改：

1. 将 Matplotlib 后端改为 `Agg`，避免无图形界面的环境因 `TkAgg` 失败；
2. 将 DEAP 采样率说明明确为 Python 预处理版的 128 Hz；
3. 删除“DE 仍是伪实现”的过时注释，保留当前 Butterworth 滤波和高斯差分熵计算；
4. 新增 `stimulus_window_starts`，显式跳过 3 秒基线，只对后 60 秒刺激期切窗；
5. 对受试者文件排序，减少文件系统枚举顺序造成的不确定性；
6. 缓存文件名从循环索引改为源文件中的真实受试者编号，例如 `s01.dat` 对应 `subject_01.npz`。

影响：标准 1 秒配置从每次试验得到 60 个刺激期窗口，首个起点为 384，末个起点为 7936。

已验证：`validate_setup.py` 对首末窗口、窗口数量和真实特征形状检查通过。  
未验证：全量 32 人缓存尚未在本记录中完成。

### `GTE_train.py`

本地修改：

1. 文档说明改为“DEAP 路线可用，SEED 仍需验证”；
2. 将命令行参数 `--ffn_dim` 真正传入 `GTENet`；
3. 在训练前创建模型输出目录；
4. 训练和验证 Accuracy 改为 `正确预测总数 / 样本总数`；上游实现平均了逐批累计准确率，会受到批次顺序影响；
5. 移除不再使用的累计变量。

已验证：脚本可导入、命令行参数可解析。  
未验证：1 epoch、早停、检查点保存和 `--resume_from` 尚未完成端到端验证。

### `model_optimized.py`

本地修改：

1. `TransformerEncoderLayer` 显式设置 `batch_first=True`；
2. 删除前向传播中进入和离开 Transformer 的两次 `permute`，统一使用 `(batch, sequence, feature)`。

目的：明确张量布局，并保持后续时序注意力池化沿序列维工作。

已验证：真实 DEAP 窗口前向输出形状为 `(1, 2)`。  
重要边界：当前主训练入口使用默认 `n_slices=1`，尚未验证多窗口序列建模。

### `README.md`

上游简要介绍已改写为面向本地复现的一站式中文说明，新增：

- 当前复现状态；
- 已验证的数据形状；
- Windows 环境和分阶段运行命令；
- 上游报告值与本地结果的区分；
- 窗口级划分、随机性、SEED 和 `n_slices=1` 等限制；
- 数据与代码许可提醒。

## 新增文件

| 文件 | 用途 | 当前验证状态 |
| --- | --- | --- |
| `CONTEXT.md` | 项目目标、关键决策、当前状态和下一步 | 已与当前验证状态核对 |
| `requirements.txt` | 声明 Python 依赖范围 | 当前 `.venv` 可导入全部依赖 |
| `validate_setup.py` | 只读检查数据、窗口、DE 特征和模型前向 | 已通过 |
| `docs/START_HERE_CN.md` | 最短上手路线 | 命令与当前入口一致 |
| `docs/DATA_LAYOUT_CN.md` | 数据目录与用途说明 | 已核对 32 个 DEAP 文件 |
| `AGENTS.md` | AI/协作者工作边界和同步规则 | 文档检查通过 |
| `docs/REPRODUCTION_PROTOCOL.md` | 冻结实验定义和验收门槛 | 阶段 A 已定义，阶段 B 待实现 |
| `docs/EXPERIMENT_LOG.md` | 只记录真实检查和训练 | 已记录快速验证，训练待运行 |
| `docs/UPSTREAM_CHANGES.md` | 本文件，追踪相对上游修改 | 根据当前 diff 整理 |

## 尚未修改或尚未解决

- `train_regularized.py` 和 `train_resumable.py` 保留上游状态，不作为当前标准入口；
- `SEEDDataset` 的文件解析、试验对应和标签排列尚未验证；
- 训练数据仍为窗口级 `random_split`，尚未实现受试者级隔离；
- Python、NumPy、PyTorch 和数据增强随机性尚未统一固定；
- 主训练脚本尚未计算 Macro-F1 和混淆矩阵；
- `--resume_from` 对文件名和检查点结构存在额外假设；
- 当前仓库没有标准 `LICENSE` 文件，不能推断为某种开源许可证。

## 更新规则

### 2026-10-15 训练兼容修复与小数据检查

- 移除从头训练和续训的 ReduceLROnPlateau 调用中的 `verbose=False`，兼容服务器 PyTorch 2.14.1+cpu。
- 修复时间缩放增强：linear 插值直接接受 `(batch, channel, time)`，补零沿最后一维进行。
- 新增可选 `--subject_ids`、`--max_samples`，默认全量行为不变；小数据使用 seed 42 抽样。
- 从头训练的初始最佳验证 Accuracy 设为 -1，确保第一次验证为 0 时也能保存检查点与曲线。
- 新增 `validate_training_runtime.py` 检查两个真实调度器调用及时间缩放两条分支；
  `validate_smoke_outputs.py` 用真实小数据验证检查点重载、优化器更新、有限 history、曲线及验证 Accuracy。
  其 `--full` 选项可对全量 1 epoch 的检查点执行同样核对。
- 新增 `jobs/gte_repair_smoke_cpu.slurm`，以小数据成功作为全量 1 epoch 的前置条件。
- 本次不验证 `--resume_from` 的其他已有假设；实际实验状态见实验日志。
- 本次验证完成：服务器两个 scheduler 调用与增强分支回归通过；320 窗口及全量 76,800 窗口各 1 epoch 成功；检查点重载后对 64/15,360 个验证窗口的 Accuracy 均与保存记录完全一致。核对脚本为 jobs/gte_verify_cpu.slurm。
- 用户授权后新增 jobs/gte_deap_50_cpu.slurm，按冻结配置从头运行 50 epochs；本次启动不修改研究代码，使用独立输出目录及代码校验和记录。

### 2026-10-13 文档目录整理

将学习目标、计划、进度、教学偏好、资源、数据说明、实验协议、实验日志、上手路线、上游变更说明和两份学习笔记/指南共 12 个 Markdown 文件移到 `docs/`。根目录保留 `README.md`、`AGENTS.md`、`CONTEXT.md`；同步文档引用、README 目录树和课程 HTML 链接。研究代码、数据目录、实验命令与历史实验结果未变。本次只检查文档路径和链接，不运行训练。

每当本地行为与上游发生新的实质差异时：

1. 在对应小节记录修改内容和目的；
2. 分别写明已验证和未验证范围；
3. 若改变实验定义，同步更新 `docs/REPRODUCTION_PROTOCOL.md`；
4. 若已经实际运行，同步追加 `docs/EXPERIMENT_LOG.md`；
5. 形成提交后，把本文件顶部的本地状态更新为提交号或标签。



### 2026-10-16 续训修复

用户授权从原有模型续训至总计 100 轮。本次只修复训练状态恢复，不改模型、数据、标签、特征或 Accuracy 定义。

- 移除依赖完整路径下划线位置解析 Accuracy、对整个 checkpoint 再次 load_state_dict、静默回退从头训练的逻辑。
- 从 checkpoint 字段恢复状态；使用 range(start_epoch, args.epochs)，将 epochs 明确定义为总轮数。
- 保持 CLI 增强概率；保留优化器和 scheduler 已存参数。源缺失、history 长度错误、目标轮数无效时直接报错。
- 无新最佳时也能输出曲线；独立保留起点副本与每轮最新 checkpoint，避免覆盖源文件。
- validate_smoke_outputs.py 只选择带训练/验证成绩的最佳模型，排除新增加的 last/source 文件；仍仅用于 1 epoch 核对。
- 新增 validate_resume_training.py，调用真实 train()、替换小型测试数据/模型，检查轮数、history、optimizer、scheduler、增强配置、无新最佳和缺失源文件分支；测试不是 EEG 性能证据。
- 新增 validate_resume_real.py，在调度节点用 32 个真实 DEAP 窗口和 GTENet 检查 1→2 轮续训；正式源检查点另核对 epoch/history/best_val_acc 元数据，不重新预测其完整验证集。
- 新增 jobs/gte_deap_resume100_cpu.slurm；先通过检查再正式续训，全程使用调度资源及冻结代码副本。

已实际运行的检查与作业状态见实验日志；旧 checkpoint 没有随机数状态，不保证与不中断训练完全相同。


### 2026-10-19 独立副本的自适应早停尝试

新增冻结训练副本 runs/adaptive_code_20261019_v1/：用training_control.py把“实质改善”的等待计数与最佳权重保存分开；小改善累计到0.001才重置40轮等待。旧源存档不含新规则时以历史best初始化，新存档保存规则状态以便后续恢复。每轮原子保存last，SIGUSR1仅请求当前轮完成后保存退出。根目录GTE_train.py、模型和数据处理不修改。新增回归包括真实train()早停，以及Linux进程收SIGUSR1后的轮次/存档检查。正式配置及命令见协议第11节；实际运行证据见实验日志。

### 2026-11-06 GitHub 发布准备

扩充 `.gitignore`，排除 IDE 配置、Python 缓存、数据/模型产物、环境变量文件、论文 PDF 与个人私有笔记。发布分支使用当前工作树的干净快照，不继承包含 IDE 连接配置的旧提交历史；保留本地旧历史。没有改变研究代码、数据定义或训练配置。

# GTE-Net 实验记录

本文件只记录实际执行过的检查和实验。计划写入“待运行队列”，不能提前填写结果。历史记录原则上只追加；如果发现错误，新增更正条目并引用原实验 ID。

## 状态词

- `PASSED`：预先定义的检查全部通过；
- `FAILED`：运行完成但未满足检查条件；
- `INTERRUPTED`：因人工停止、资源限制或外部原因未完成；
- `PLANNED`：尚未运行，不属于实验结果。

## 汇总

| 实验 ID | 日期 | 类型 | 数据划分 | 状态 | 主要结论 |
| --- | --- | --- | --- | --- | --- |
| VAL-20261010-01 | 2026-10-10 | 环境/数据/前向检查 | 单个真实 DEAP 窗口 | PASSED | 32 个文件、数据形状、窗口、DE 特征和前向输出通过 |
| SMOKE-DEAP-01 | — | 1 epoch 冒烟训练 | 窗口级 80/20 | PLANNED | 尚未运行 |
| TRAIN-DEAP-50-01 | — | 50 epochs 中程训练 | 窗口级 80/20 | PLANNED | 尚未运行 |
| TRAIN-DEAP-700-01 | — | 700 epochs 上游设置 | 窗口级 80/20 | PLANNED | 尚未运行 |

2026-10-16 更新索引：上表的 PLANNED 为初始计划快照，保留历史；1 epoch 和 50 epochs 的实际执行使用下列服务器实验 ID，最新状态以对应完成补记为准。

| 实际实验 ID | 完成日期 | 最新状态 | 主要结果 |
| --- | --- | --- | --- |
| FULL-COLLEGE-1887643 / 核对 1887644 | 2026-10-15 | PASSED | 全量 1 epoch，窗口级验证 56.76%，重载核对一致 |
| TRAIN50-COLLEGE-1887646 | 2026-10-15 | 训练 COMPLETED；最佳模型待独立重载核对 | 50 epochs，第 48 轮最佳窗口级验证 64.93%；单次结果 |
| RESUME100-COLLEGE-1887720 | 2026-10-16 | COMPLETED、0:0 | 总计100轮，第98轮最佳窗口级验证69.9349%；续训分支 |
| ADAPTIVE-COLLEGE-1887873 | 2026-10-20 | COMPLETED、0:0；early_stopping | 总计378轮，第338轮最佳窗口级验证75.4883%；未独立重载复核 |

## VAL-20261010-01

### 目的

确认当前 Python 环境能够读取真实 DEAP 数据，对一个真实 EEG 窗口完成基线校正、五频带 DE 计算和 GTE-Net 前向传播。

### 代码状态

- 上游基线 HEAD：`a6facc96b2c159abe14b8d88faf95564d2634122`
- 工作树：包含未提交的本地复现修改，属于 dirty working tree；具体修改见 `docs/UPSTREAM_CHANGES.md`

### 环境

```text
python=3.11.9
torch=2.13.0+cpu
numpy=2.4.6
scipy=1.17.1
sklearn=1.9.0
matplotlib=3.11.1
tqdm=4.70.0
cuda_available=False
cuda_version=None
```

### 命令

```powershell
.\.venv\Scripts\python.exe validate_setup.py
```

### 实际输出

```text
OK: 32 DEAP files
OK: data=(40, 40, 8064), labels=(40, 4)
OK: 60 one-second stimulus windows, baseline excluded
OK: real EEG features time=(32, 128), freq=(32, 5)
OK: model forward output=(1, 2)
PyTorch device available: cpu
```

### 结论与边界

状态：`PASSED`。

这证明当前环境下单文件读取、一个真实窗口的特征计算和模型前向传播可用。它不证明：

- 全量 76,800 个窗口预处理已经完成；
- 1 epoch 训练链路已经完成；
- 检查点或训练曲线能够正确保存；
- 上游约 80.44% 已被复现；
- 模型具有跨受试者泛化能力。

## VAL-20261014-01

### 目的与证据

用户亲自在项目 PowerShell 中运行环境、数据和真实窗口前向检查，并在聊天中提交终端截图。以下为截图中的实际输出，助手本次未代跑。

### 代码状态与环境

- 本次核对 Git HEAD：`a6facc96b2c159abe14b8d88faf95564d2634122`；工作树包含未提交的本地修改（dirty working tree）。截图未记录运行瞬间的 diff，不能据此完整还原运行版本。
- 截图显示 PyTorch 使用 `cpu`；本次未提供 Python 和依赖版本，未沿用历史版本作为本次实测值。
- 数据集与任务：DEAP Valence 二分类路线的检查；检查未进行训练/验证集划分。

### 完整命令

```powershell
cd D:\workspace\GTE-Net
.\.venv\Scripts\python.exe validate_setup.py
```

### 实际输出

```text
OK: 32 DEAP files
OK: data=(40, 40, 8064), labels=(40, 4)
OK: 60 one-second stimulus windows, baseline excluded
OK: real EEG features time=(32, 128), freq=(32, 5)
OK: model forward output=(1, 2)
PyTorch device available: cpu
```

### 结论与边界

状态：`PASSED`（依据用户提供的终端截图，命令完成并返回 PowerShell 提示符）。32 个数据文件及抽样数据、60 个刺激期窗口、单个真实窗口特征和模型前向检查通过。耗时未提供；本次没有训练 Accuracy 或 Macro-F1，不证明全量缓存、训练、检查点保存或跨受试者泛化已经完成。`SMOKE-DEAP-01` 仍为待运行。

## 待运行队列

### SMOKE-DEAP-01

- 目标：完成首次全量缓存和 1 epoch 训练；
- 命令：使用 `docs/REPRODUCTION_PROTOCOL.md` 第 7.3 节冻结命令；
- 验收：阶段 A 的 Gate A1 和 A2；
- 当前状态：`PLANNED`。

### TRAIN-DEAP-50-01

- 前置条件：`SMOKE-DEAP-01` 通过；
- 目标：检查 50 epochs 的收敛趋势和资源消耗；
- 当前状态：`PLANNED`。

### TRAIN-DEAP-700-01

- 前置条件：50 epochs 运行合理，并完成随机性与资源评估；
- 目标：按冻结配置与上游约 80.44% 并列比较；
- 当前状态：`PLANNED`。

## 新实验模板

复制以下模板并追加到文件末尾：

````markdown
## <实验 ID>

### 目的

### 代码状态

- Git HEAD：
- `git status --short`：

### 环境

- Python：
- PyTorch：
- CUDA / 设备：

### 数据与划分

- 数据目录：
- 缓存目录：
- 划分单位与随机种子：

### 完整命令

```powershell
# command
```

### 实际结果

- 状态：
- 耗时：
- 样本数：
- 最佳 epoch：
- Accuracy：
- Macro-F1：
- 输出文件：

### 异常与偏差

### 结论与边界
````


## SMOKE-COLLEGE-1887465：2026-10-15 补记失败终态

- 任务：DEAP Valence 二分类，32 名受试者、76,800 窗口，窗口级 80/20，split seed 42，计划 1 epoch。
- 命令：服务器根目录 `sbatch jobs/gte_smoke_cpu.slurm`；原脚本与提交记录保存在服务器 jobs/、manifests/submission-1887465.md。
- 版本：基线 HEAD a6facc96b2c159abe14b8d88faf95564d2634122 的 dirty working tree 副本；训练入口 SHA256 2dde3095a127a6dc33c64cd5bc97b27e82769d545d9de319ae7f8eaa54da3efb。
- 实时查询 sacct：2026-10-14 19:41:00 开始，21:47:29 结束，02:06:29，FAILED，ExitCode 1:0。
- 环境：Python 3.11.15，torch 2.14.1+cpu，c001，8 CPU，10G。
- 已通过：validate_setup.py；32 名受试者预处理（约 1:59:00）及全部 76,800 样本加载。
- 失败：GTE_train.py:301，ReduceLROnPlateau 不接受 verbose；未进入训练 epoch，无 Accuracy、Macro-F1、模型或曲线。
- 证据：服务器 logs/slurm-1887465.out、logs/slurm-1887465.err、runs/smoke/1887465/environment.txt。

## FIX-COLLEGE-20261015：回归检查与顺序作业提交

- 用户授权：修复后先少量真实数据训练验证，再处理全部数据。
- 本地仍为上述 HEAD 的 dirty working tree，保留所有既有未提交改动。
- 已复现增强错误：强制时间缩放报 `NotImplementedError: Got 4D input, but linear mode needs 3D input`。
- 修复：两个 scheduler 调用移除 verbose；时间插值和补零维度修复；首次验证准确率为 0 也保存检查点。
- 本地实际检查：validate_training_runtime.py 通过（两个真实 scheduler 调用、缩短/拉长、输入不变）；三个新增/修改 Python 文件 py_compile 通过；git diff --check 无错误（仅既有 CRLF 提示）。
- 顺序作业：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch jobs/gte_repair_smoke_cpu.slurm`，返回 1887641；已观察 RUNNING，节点 c001。
- 完整训练配置见仓库 jobs/gte_repair_smoke_cpu.slurm：小数据 subject_ids=1、max_samples=320，256/64 窗口；随后全量 61,440/15,360；各 1 epoch，其余冻结参数不变。
- 训练入口 SHA256：7ec4125991fe9802fc6be25fc795d28bd8901f88476ebdc4f1211b6dc4c2bb85；完整代码与依赖清单位于服务器 runs/repair/1887641/。
- 产物隔离：runs/repair/1887641/small/、full/；日志 logs/slurm-1887641.out、.err；原始数据与既有缓存未修改。
- 此条只记录已执行检查与提交，不代表小数据或全量完成；终态和实际指标另行追加。Macro-F1 未输出。

### FIX-COLLEGE-20261015 提交更正

- 1887641 停在依赖清单采集（pip freeze），尚无训练输出；为避免非必要采集阻塞，已取消此作业。
- 作业脚本改为复制昨天的依赖快照，文件名明确为 environment-snapshot-20261014.txt；运行时由回归检查打印实际 torch 版本，不把旧清单冒充新采集。
- 同一顺序作业重新提交为 1887643，已观察 RUNNING，节点 c002。训练入口 SHA256 不变；新回归脚本校验和由 runs/repair/1887643/code.sha256 保存。
- 输出及日志中的作业号均改为 1887643。本条仍不代表训练完成。
- 服务器最小调度器复现也已返回预期的 verbose TypeError（原调用），与原始完整作业报错一致。

### 1887643 运行中核对

- 计算节点实际输出 torch 2.14.1+cpu；两个调度器真实调用、时间缩放缩短/拉长分支全部通过。
- 启动阶段曾多次处于 D（disk sleep），打开的文件包括 SciPy 扩展库；后续 validate_setup.py 已打开原始 s01.dat。此为当前观测，不据此认定具体存储故障。
- 依赖打包诊断未完成，已停止 tar，残留保留为服务器 runs/repair-runtime/site-packages.partial.tar，不用于运行。
- 本地也检查了已有缓存：仅 subject_01 至 subject_08，mtime 均为 2026-10-14。一次本地备用训练启动触发全量预处理后立即停止；未生成新缓存，现有八个文件时间与大小保持不变。

### SMALL-COLLEGE-1887643：小数据流程 PASSED

- 2026-10-15 17:41:23 前完成；在 c002、torch 2.14.1+cpu、8 CPU、10G 上单次运行。
- 数据：subject_01 的 2,400 窗口缓存中以 seed 42 随机选取 320；窗口级 seed 42 的 80/20 划分为 256 训练、64 验证，Valence <5/≥5 二分类。
- 参数与完整命令：jobs/gte_repair_smoke_cpu.slurm 的 COMMON 数组，加 --subject_ids 1 --max_samples 320 --model_save_path "$RUN_DIR/small/GTE_best.pth"；提交命令与代码状态同上。
- 真实数据检查通过：32 文件，原始 (40,40,8064)/(40,4)，60 个一秒刺激窗口、排除基线，特征 (32,128)/(32,5)，真实前向 (1,2)。
- 1 epoch：Train Loss 0.7165，Train Accuracy 0.47265625；Val Loss 0.6992，Val Accuracy 0.4375。Accuracy 按整个对应集合计算；Macro-F1 未计算。
- 检查点重新加载成功，优化器存在更新状态、history 为有限值、曲线文件非空；重新对全部 64 个验证窗口预测，Accuracy=0.43750000，与保存的指标完全一致。
- 产物：runs/repair/1887643/small/GTE_best_train_acc_0.47265625_val_acc_0.4375.pth（3,372,457 bytes）及对应 _final_curves.png（32,402 bytes）。
- SMALL_PASSED 后于 17:41:23 进入 FULL_START，加载 76,800 窗口；已观察 61,440/15,360 划分及训练批次进度，无全量终态。
- 小数据数值仅验证代码链路，不是泛化性能或上游报告值的复现。

## FULL-COLLEGE-1887643：全量 1 epoch PASSED

- 2026-10-15：SMALL_PASSED 后 17:41:23 启动全量，17:45:17 输出 FULL_PASSED；全量阶段约 3 分 54 秒（含加载、训练、验证、保存）。
- 调度器：1887643 主作业与 batch 均 COMPLETED，ExitCode 0:0，总耗时 00:15:08；之前约 11 分钟主要为启动、依赖/数据读取与前置检查，不等同于训练耗时。
- sacct 中 1887643.3 的 124 为独立 timeout 15 依赖导入诊断步骤，非训练失败；主作业、batch 和 SMALL/FULL 门槛均成功。
- 环境：c002，Python 3.11.15，torch 2.14.1+cpu，8 CPU、10G，具体依赖参考带日期的旧快照，不冒充新采集。
- 代码：基线 a6facc96b2c159abe14b8d88faf95564d2634122 的 dirty working tree；训练入口 SHA256 7ec4125991fe9802fc6be25fc795d28bd8901f88476ebdc4f1211b6dc4c2bb85；训练时完整清单 runs/repair/1887643/code.sha256。
- 完整命令：`sbatch jobs/gte_repair_smoke_cpu.slurm`，完整 COMMON 参数与顺序流程已保存在 runs/repair/1887643/submitted-job.slurm；脚本 SHA256 e53043d377bbd4752fdcc7a2908c13d22380aacb1b361340b8e0432569d4de04。
- 数据/任务：DEAP Python 预处理数据，32 受试者，Valence <5/≥5 二分类，全部 76,800 个一秒窗口；窗口级 random_split seed 42，训练 61,440，验证 15,360；其他随机源未统一固定。未改原始数据、标签、特征定义或划分单位。
- 1 epoch 单次结果：Train Loss 0.6895，Train Accuracy 0.556982421875；Val Loss 0.6863，Val Accuracy 0.567578125。Accuracy 分母为完整对应集合；Macro-F1 未计算。
- 模型：runs/repair/1887643/full/GTE_best_train_acc_0.556982421875_val_acc_0.567578125.pth；曲线为同名前缀 _final_curves.png。
- 结论：全量训练、验证及输出保存完成；不代表多次均值、受试者独立评估或复现上游 80.44%。
- 额外核对已提交：`sbatch --nodelist=c002 jobs/gte_verify_cpu.slurm 1887643` → 1887644。使用扩展后的 validate_smoke_outputs.py --full，对全部 15,360 个验证窗口重新预测；验证脚本的新 SHA256 在该核对作业日志中另行记录，训练代码未变。此时独立核对尚无终态。

### FULL-COLLEGE-1887643：独立重载核对 PASSED（1887644）

- 作业 1887644：COMPLETED、ExitCode 0:0、耗时 21 秒。
- 完整命令：`sbatch --nodelist=c002 jobs/gte_verify_cpu.slurm 1887643`；该脚本调用 `.venv/bin/python validate_smoke_outputs.py --run_dir runs/repair/1887643/full --full`，8 CPU、10G、CPU、30 分钟上限。
- 验证脚本 SHA256：7dcee855292a3a2d4b7b73427601f3f9baca53ed6e20336a8a10fa2cc0bb8fbc。
- 实际结果：读取 32 名受试者的 76,800 个窗口，以原 split seed 42 重建验证集；重载检查点，对全部 15,360 个验证窗口预测，Accuracy=0.567578125，与 best_val_acc 及 history.val_acc 的差均小于 1e-12。
- 同时确认：优化器有更新状态、epoch=0、history 全部有限、曲线文件非空、验证输入/输出均有限、模型 state_dict 严格加载成功。
- 本次已完成用户要求的“修复 → 少量真实数据训练验证闭环 → 全量数据训练验证闭环”；未运行 50/700 epochs，未验证跨受试者性能与续训。

## TRAIN50-COLLEGE-1887646：50 epochs 提交与实际启动

日期：2026-10-15。用户在确认先跑 50 epochs 后明确要求“那开始吧”。

- 提交命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 jobs/gte_deap_50_cpu.slurm`；返回 Submitted batch job 1887646。
- 已执行检查：bash -n 通过；sbatch --test-only 接受申请（测试编号 1887645，不是实际训练作业）；正式提交后观察 1887646 RUNNING、c002，日志已进入 Epoch 1/50 [Train]，约 9 batches/s。
- 数据与任务：全部 32 名受试者、76,800 个 DEAP 一秒窗口，Valence <5/≥5 二分类；窗口级 80/20 random_split seed 42，61,440 训练、15,360 验证。其他随机性尚未统一固定。
- 配置：从头训练，不传 --resume_from；仅 epochs=50 和输出目录区别于已通过的全量 1 epoch。batch_size=32、lr=1e-4、weight_decay=1e-4、time_window=1、gat_heads=2、transformer_heads=4、transformer_layers=1、d_model=128、ffn_dim=512、data_augmentation=0.3、early_stopping_patience=100；模型默认 n_slices=1。完整调用保存在 jobs/gte_deap_50_cpu.slurm 和运行目录 submitted-job.slurm。
- 资源：CPU 分区，c002，1 节点、1 任务、8 CPU、10G、24 小时上限；沿用已验证的 .venv（最近实测 Python 3.11.15、torch 2.14.1+cpu），未重装依赖。
- 代码状态：本地 HEAD a6facc96b2c159abe14b8d88faf95564d2634122，dirty working tree。保留全部原有未提交改动；本次仅新增作业脚本和更新指导文档，不修改训练、模型或数据处理代码。远程为该工作树的非 Git 副本。
- 训练入口 SHA256：7ec4125991fe9802fc6be25fc795d28bd8901f88476ebdc4f1211b6dc4c2bb85；DatasetProcessSave.py：30260a6b543a16df131571bff792fb718fc58babc10cae2e4a73409cd049ab38；model_optimized.py：1f3c03af0401b4784ce4e35412eb4daa5e3286c7eca04579ccc30d2951a069ca。提交前与已通过的服务器代码一致。
- 作业脚本 SHA256：fb8edc1082c6a1821c6a0bc463dffe0e2eaeca434b3be96369daeb9a77db5893。
- 输出：runs/deap_window_split_50/1887646/；日志 logs/slurm-1887646.out、logs/slurm-1887646.err。保留 submitted-job.slurm、code.sha256 与明确标注日期的 environment-snapshot-20261014.txt（旧快照，不是新采集）。
- 当前结论：训练已实际启动，尚无 50 epochs 终态、最佳 epoch、最终 Accuracy 或收敛判断。Macro-F1 未输出；后续结果仍属单次窗口级评估，不能解释为跨受试者性能。
- 根据已通过的 1 epoch 及当前批次速度估计约 3–4 小时，非完成时间保证；本次不启动 700 epochs。

## TRAIN50-COLLEGE-1887646：2026-10-16 完成补记与工作状态

### 实际完成情况

- 2026-10-16 再次通过 SSH 查询 sacct：1887646 主作业和 batch 均 COMPLETED，ExitCode 0:0；2026-10-15 18:18:54 开始、21:22:08 结束，耗时 03:03:14。查询时 squeue 无本账号运行/排队作业。
- 日志包含完整的第 1 至 50 轮及 FULL_50_PASSED。已把标准输出保存到本地 runs/deap_window_split_50/1887646/slurm-1887646.out；用只读解析核对 50 个连续 epoch，日志中的 Train/Val Loss、Accuracy 均为有限值。
- 数据、命令和代码状态沿用上文启动记录：DEAP Valence 二分类，32 名受试者、76,800 个一秒窗口；窗口级 80/20、split seed 42，训练 61,440、验证 15,360，模型默认 n_slices=1。其他随机源未统一固定。
- 完整命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 jobs/gte_deap_50_cpu.slurm`。脚本内含全部冻结参数，运行时副本为 runs/deap_window_split_50/1887646/submitted-job.slurm；未传 --resume_from，从头训练。
- 版本：基线 HEAD a6facc96b2c159abe14b8d88faf95564d2634122 的 dirty working tree 副本；运行代码清单已重读。GTE_train.py SHA256=7ec4125991fe9802fc6be25fc795d28bd8901f88476ebdc4f1211b6dc4c2bb85，数据处理与模型校验和同提交记录。本次仅记录、核对与更新文档，不修改研究代码。
- 资源环境沿用提交记录：c002、8 CPU、10G、CPU 版 torch 2.14.1+cpu，Python 3.11.15；依赖文件仍是明确标注日期的 2026-10-14 快照，未重新采集。

### 单次结果与已确认产物

| 位置 | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 第 1 epoch，日志值 | 0.6886 | 0.5588 | 0.6858 | 0.5676 |
| 第 48 epoch，最佳验证轮，日志值 | 0.6446 | 0.6437 | 0.6406 | 0.6493 |
| 第 50 epoch，日志值 | 0.6428 | 0.6458 | 0.6407 | 0.6434 |

- 最佳模型文件名中的完整精度：Train Accuracy=0.6437174479166666、Val Accuracy=0.6492838541666667，即窗口级验证 64.93%。日志保留四位小数；该完整值尚未通过此次模型的独立重载重新预测核对。
- Accuracy 对应每轮整个训练/验证集合，不是独立测试集结果。仅一次运行，无多次均值/标准差；Macro-F1 未计算。
- 最佳模型：服务器 runs/deap_window_split_50/1887646/GTE_best_train_acc_0.6437174479166666_val_acc_0.6492838541666667.pth，3,380,081 bytes，文件存在且非空。
- 训练曲线：同名前缀 _final_curves.png，109,153 bytes，文件存在且非空。本次未对曲线图像进行视觉审阅。
- 已验证事实：50 epochs 全部结束、调度器成功退出、日志指标有限、保存产物存在。
- 初步观察：训练和验证损失较首轮总体下降，验证准确率较首轮上升，但第 49/50 轮有回落；不能仅凭这些点认定稳定收敛。

### 当前结论与最小下一步

当前可表述为：已跑通 GTE-Net 的 DEAP Valence 二分类全量训练流程，并完成 50 epochs 窗口级复现实验；整个项目复现尚未完成。

- 已完成：真实窗口前向、32 名受试者缓存、小数据与全量 1 epoch 训练验证及独立重载核对、50 epochs 训练验证与模型/曲线保存。
- 优先下一步：独立重载核对这次 50 epochs 的最佳检查点，再审阅曲线与上游配置差异。现有 validate_smoke_outputs.py 假设 1 epoch 且目录只有一个检查点，不能直接用于本次多检查点目录；开展时需调整核对入口。
- 仍未完成：700 epochs、统一随机种子及重复运行、上游约 80.44% 的性能复现、受试者级独立评估、Macro-F1/混淆矩阵、SEED 路线和断点续训验证。
- 评估边界：当前同一受试者和相邻窗口可能跨训练/验证集；窗口级 64.93% 不能解释为跨受试者性能。上游 80.44% 始终标为上游报告值。
- 本次不启动新训练，不改变教学掌握程度；助手代跑不计入用户学习进度。已有未提交改动保留，未进行 Git 提交或推送。


## RESUME100-COLLEGE-1887718：续训修复、提交与排队

日期：2026-10-16。用户要求继续使用学校集群、遵守调度规则，并明确选择“接着之前的模型练，到总计 100 轮（需先修复续训）”。

### 本地已执行检查

- 修复前运行 `.\.venv\Scripts\python.exe validate_resume_training.py`：实际 train() 续训路径在 `args.resume_from.split('_')[6]` 抛出 IndexError；路径中的下划线位置不可靠。
- 修复后同一命令通过，输出 RESUME_REGRESSION_PASSED。测试用 10 个合成样本和 TinyModel（不是 DEAP 性能实验），验证从 epoch=0 接到 epoch=1、history 长度 2、AdamW step=2、scheduler 计数继承、增强概率不被覆盖、源文件不变、无新最佳和缺失源路径。
- `.\.venv\Scripts\python.exe validate_training_runtime.py` 通过，实际输出 torch 2.13.0+cpu；两个 scheduler 调用和时间缩放缩短/拉长分支通过。
- `.\.venv\Scripts\python.exe -m py_compile GTE_train.py validate_resume_training.py validate_resume_real.py validate_smoke_outputs.py` 通过。
- 本次修复及参数定义见 UPSTREAM_CHANGES.md、REPRODUCTION_PROTOCOL.md 第 10 节。没有运行本地 DEAP 训练或重新预测 50 轮最佳模型的完整验证集。

### 实际提交

- 远程先检查 squeue，未发现该账号其他运行/排队作业；重读服务器 AGENTS.md。Slurm cpu 分区可用，单节点内存上限 20G；本次沿用 8 CPU、10G、24 小时，不指定节点。
- 服务器原 GTE_train.py、DatasetProcessSave.py、model_optimized.py 校验和与上次训练记录相符。本次冻结副本上传至 `runs/resume100_code_20261016_v1/`，未覆盖服务器原训练入口或前次产物。
- 冻结目录全部 8 个文件 sha256sum -c 通过，bash -n 通过；sbatch --test-only 接受（测试编号 1887717，不是训练作业）。
- 完整提交命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch runs/resume100_code_20261016_v1/gte_deap_resume100_cpu.slurm /share/home/ycg_luanjingjie/GTE-Net/runs/resume100_code_20261016_v1`。
- 实际返回 `Submitted batch job 1887718`；第一次查询为 PENDING，原因 Priority。此时尚无运行日志、检查终态或训练结果。
- 代码状态：HEAD a6facc96b2c159abe14b8d88faf95564d2634122，dirty working tree；保留原有改动。GTE_train.py SHA256=f92c0221b4f79665a57e12d8472ddc515c8fa5983253c783eb63aec89dc79d9f；数据处理与模型校验和仍为上次值，完整清单见冻结目录 code.sha256。远程目录是非 Git 副本。
- 作业入口内使用 srun 执行 Python：续训回归、scheduler/增强检查、32 个真实 DEAP 窗口（26 训练/6 验证）的 1→2 轮续训检查；仅全部通过后正式训练。登录节点没有运行 Python 训练或模型检查。
- 正式数据/任务：DEAP Valence <5/≥5 二分类；32 受试者、76,800 个一秒窗口；原窗口级 random_split seed 42，61,440 训练/15,360 验证。其他随机性未统一固定。
- 源模型：`runs/deap_window_split_50/1887646/GTE_best_train_acc_0.6437174479166666_val_acc_0.6492838541666667.pth`（第 48 轮）；本次计划运行第 49–100 轮，共 52 轮，不是从第 50 轮末尾接续。
- 训练参数沿用 50 轮，epochs=100 为总目标；batch_size=32、lr=1e-4（恢复优化器已存值）、weight_decay=1e-4、time_window=1、gat_heads=2、transformer_heads=4、transformer_layers=1、d_model=128、ffn_dim=512、data_augmentation=0.3、early_stopping_patience=100；n_slices 默认 1。
- 输出：`runs/deap_window_split_resume100/1887718/`，日志 `logs/slurm-1887718.out`、`.err`；作业保存脚本、代码/源检查点校验和及带日期的旧依赖快照（不冒充新采集）。
- 本次是旧模型的续训分支，不是独立重复运行；旧 checkpoint 未存 RNG 状态，不保证与不中断训练逐步一致。Accuracy 仍按完整对应集合计算，无新 Macro-F1 或跨受试者结果。


### 1887718：调度分配确认

- 2026-10-16 20:29:04（北京时间）Slurm 开始作业；scontrol 确认 RUNNING、c001、NumCPUs=8、NumTasks=1、CPUs/Task=8、mem=10G、TimeLimit=24 小时、QOS=normal。
- 作业日志再次通过冻结代码清单全部校验；20:29:27 打印主机 c001，已启动 srun Python 前置检查。当前没有前置检查通过或正式第 49 轮训练的日志，不将 RUNNING 当作训练已成功。
- `git diff --check -- GTE_train.py README.md CONTEXT.md docs/REPRODUCTION_PROTOCOL.md docs/UPSTREAM_CHANGES.md docs/EXPERIMENT_LOG.md validate_smoke_outputs.py` 退出码 0（仅 Git 行尾提示）。


### 1887718：启动读取等待后取消，改申请 c002

- 计算节点检查通过 Slurm 原作业内的短诊断步骤执行（srun --jobid=1887718 --overlap --ntasks=1 --cpus-per-task=1），未绕过调度登录计算节点。观察 Python 状态 D，打开文件先为 torch/distributed 的 pyc，后为标准库 pickletools.pyc；sstat 显示 CPU 累计约 1 秒、内存约 105 MB。仅能确认依赖读取等待，不据此认定存储故障。
- 等待约 9 分钟仍未输出 RESUME_REGRESSION_PASSED、真实数据检查或正式训练起点。c002 查询为空闲，且此前 50 轮在该节点完成；决定改由调度申请 c002，不能保证换节点就解决读取问题。
- `sbatch --test-only --nodelist=c002 ...` 接受，测试编号 1887719；随后 `scancel 1887718`。sacct 确认主作业及 Python 检查步骤 CANCELLED，Elapsed 00:09:06；没有开始真实 DEAP 检查或正式续训。原目录和日志保留，不作为训练结果。


## RESUME100-COLLEGE-1887720：同一冻结配置改由 c002 调度运行

日期：2026-10-16。沿用上一节用户授权、数据、源模型、训练配置、修复代码和检查流程，仅改为向调度器申请 c002；旧作业 1887718 的 Python 检查步骤已取消，没有两份正式训练并行。

- 完整命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 runs/resume100_code_20261016_v1/gte_deap_resume100_cpu.slurm /share/home/ycg_luanjingjie/GTE-Net/runs/resume100_code_20261016_v1`。
- 实际返回 `Submitted batch job 1887720`；资源仍为 cpu、8 CPU、10G、24 小时。代码与源 checkpoint SHA256 同 1887718 的冻结配置；源 SHA256=0db652de2cba224da6e212800c2ca9be3614a2f71362cef44120c8a5ecd3ad10。
- 新输出 `runs/deap_window_split_resume100/1887720/`；新日志 `logs/slurm-1887720.out`、`.err`。不覆盖旧作业输出。
- 正式目标仍为从第 48 轮最佳模型继续第 49–100 轮，32 受试者、76,800 窗口，61,440/15,360 原窗口划分；是续训分支，非独立重复，未统一全局随机性，未计算 Macro-F1。
- 截至提交记录尚未确认前置检查或正式训练启动；后续以补记为准。


### 1887720：计算节点前置检查 PASSED，进入正式数据加载

- 2026-10-16：Slurm RUNNING、c002、8 CPU、10G；实际 Python 为既有 .venv（最近实测 3.11.15，未重新采集完整环境），本作业实际打印 torch 2.14.1+cpu。
- srun 检查步骤 .0（合成数据续训回归）COMPLETED，退出码 0:0、33 秒；.1（两个 scheduler 调用和时间缩放分支）COMPLETED，退出码 0:0、4 秒。
- .2（真实 DEAP/GTENet 续训）COMPLETED，退出码 0:0、1 分 02 秒；日志 REAL_RESUME_PASSED、RESUME_PREFLIGHT_PASSED。
- 真实小检查使用 subject_01 的 2,400 窗口中 seed 42 抽取 32 个，再 seed 42 做窗口级 26/6 划分；配置为冻结参数但只用该子集，先 epochs=1，再 --resume_from 该 last checkpoint、epochs=2。完整两个 CLI 由冻结 validate_resume_real.py 的 common 列表构造；代码 SHA256=253eaf9bce46f743f8297f1eb1c541c7dd873d7b8c166766d60ef16949d9ef0c。
- 小检查单次日志（Accuracy 为完整26训练/6验证集合，四位小数）：第1轮 Train Loss=0.7168、Train Acc=0.4615、Val Loss=0.6948、Val Acc=0.3333；续训第2轮 0.7217、0.3846、0.6999、0.3333。仅验证续训链路，不代表模型性能；Macro-F1 未计算。
- 断言确认：epoch=1、四项 history 长度2且首项保留、AdamW 所有状态 step=2、scheduler last_epoch=2、增强概率0.3、参数有更新、源文件 SHA256 不变、曲线生成。正式源模型元数据也确认 epoch=47、history长度48、best_val_acc=0.6492838541666667；未对该源模型完整验证集重新预测。
- 日志输出 FULL_RESUME_100_START，随后正式训练入口开始加载全部 DEAP 数据；本补记时尚未观察第49轮或新准确率。


### 1887720：本次交接快照

- 最后一次查询为 RUNNING、c002、Elapsed 00:04:26。stdout 已到正式数据加载 `Loading preprocessed DEAP dataset...`，尚未输出正式 `Epoch 49/100`，尚无新正式 Accuracy 或100轮终态。
- 已将当前 stdout 与 submitted-job.slurm 保存到本地 `runs/deap_window_split_resume100/1887720/`。这是读取时的快照，后续以服务器持续写入的 logs/slurm-1887720.out、.err 为准。
- 原始数据、已有缓存、前次模型、服务器原训练入口均未改动；正式作业使用冻结代码副本。没有进行 Git 提交或推送，也没有创建自动监控。


## ADAPTIVE-COLLEGE-1887873：100轮起点的自适应续训

日期：2026-10-19。用户明确同意上限2000轮、按实质改善早停，并要求开始在学校集群尝试；单次作业仍遵守Slurm资源和时间限制。

### 提交前确认与本地检查

- 实时查询squeue：账号没有运行或排队作业；旧作业1887720 COMPLETED、ExitCode0:0、耗时03:19:09；第100轮last存档存在（3,368,691 bytes）。已有本地终态日志显示第98轮best_val_acc=0.6993489583333333，第100轮日志Val Acc=0.6904，FULL_RESUME_100_PASSED；这是旧结果，不是本次新成绩。
- 修改仅在独立副本 runs/adaptive_code_20261019_v1/，由上次冻结副本派生；根目录GTE_train.py、DatasetProcessSave.py、model_optimized.py以及旧运行产物保持原样。
- `.\.venv\Scripts\python.exe runs/adaptive_code_20261019_v1/validate_early_stop.py`：先建立检查，缺失training_control时失败；实现后3项测试通过（小进步累计、耐心阈值、恢复状态/更改规则重置、NaN拒绝）。
- `.\.venv\Scripts\python.exe runs/adaptive_code_20261019_v1/validate_resume_training.py`：真实train()配合TinyModel/10合成样本，通过续训状态恢复、缺失源错误和无新最佳；patience2、目标10时在第3轮停止，last记录epoch2/history长度3/counter2，避免只测试辅助类。
- `.\.venv\Scripts\python.exe runs/adaptive_code_20261019_v1/validate_training_runtime.py`：两个scheduler调用及增强缩短/拉长分支通过，实际torch2.13.0+cpu（本地）。
- 对副本GTE_train.py、training_control.py、validate_resume_real.py、validate_resume_training.py的py_compile通过。Linux SIGUSR1路径安排在计算节点检查，本地Windows未冒充已验证该路径。
- 代码基线HEAD=a6facc96b2c159abe14b8d88faf95564d2634122，dirty working tree；保留既有未提交修改，未提交/推送。副本训练入口SHA256=beb4e95bf0bd6ec01e68a045b26ae4b48a2a8702b2bf301d039e2bd0ed270fb7，控制模块SHA256=242526cada271cbc296354fc14b67e5953245762730c4b35f1bce563b9d20366；模型和数据处理哈希与100轮冻结副本一致。全部9文件清单见副本code.sha256。

### 实际提交与配置

- 重新检查cpu分区和服务器AGENTS.md相关规则。冻结副本上传到服务器同名目录，全部sha256sum -c和bash -n通过；sbatch --test-only --nodelist=c002接受（测试编号1887872，不是训练作业）。
- 完整命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 runs/adaptive_code_20261019_v1/gte_adaptive_cpu.slurm /share/home/ycg_luanjingjie/GTE-Net/runs/adaptive_code_20261019_v1`，返回Submitted batch job1887873。
- 2026-10-19 19:00:42（北京时间）启动；scontrol确认RUNNING、c002、8 CPU、10G、cpu分区、TimeLimit24小时；所有Python由srun执行，没有在登录节点执行训练或模型检查。
- 数据任务：DEAP Valence<5/≥5二分类；32人76,800个1秒窗口；原窗口random_split seed42，61,440训练/15,360验证。保持模型默认n_slices1、batch32、lr命令初值1e-4但恢复AdamW存档lr5e-5、weight_decay1e-4、gat_heads2、transformer_heads4、transformer_layers1、d_model128、ffn_dim512、增强0.3。
- 起点：runs/deap_window_split_resume100/1887720/GTE_best_last.pth，第101轮继续到总轮数上限2000。新增早停patience40/min_delta0.001，初始参考历史最好0.6993489583333333，计数从这次新规则起点0开始；微小改善累计达阈值再重置。最佳权重仍按任意严格改善保存。完整配置见协议第11节。
- 单次24小时作业，提前约600秒USR1请求当前epoch结束后保存退出；没有自动连锁提交作业。退出原因明确区分early_stopping/max_epochs/walltime_signal；若epoch未能在余时结束，Slurm仍可能强制终止，原子last保存保护之前完整状态。
- 先运行规则测试、真实train()回归和Linux自发USR1检查、scheduler/增强检查、32真实DEAP窗口1→2续训；通过后再进入正式训练。首次查询已进入检查阶段，尚无正式第101轮或新成绩。
- 输出runs/deap_window_split_adaptive/1887873/；日志logs/slurm-1887873.out、.err。保存源文件哈希、脚本、冻结代码清单、带20261014日期的旧环境快照（不是新采集）及历史第98轮最佳权重副本。
- 仍是同一验证集的单次续训分支，不是独立重复或跨人评估；旧存档无RNG状态，无新Macro-F1或泛化结论。本条不将计划写为完成。


### 1887873：前置检查 PASSED，正式第101轮已启动

- 2026-10-19实时日志：ADAPTIVE_PREFLIGHT_PASSED、ADAPTIVE_TRAIN_START；正式加载32人76,800窗口，确认61,440/15,360、seed42。恢复日志 completed_epochs=100、target_epochs=2000、augmentation=0.3、lr=5e-05，EARLY_STOP_CONFIG patience40/min_delta0.001/anchor0.6993489583333333/bad_epochs0。
- 已观察 Epoch101/2000 [Train] 进入217/1920批次附近，约9 batches/s；此时没有本轮验证结果、早停或最终新成绩。
- Slurm .0规则检查、.1真实train回归、.2 scheduler/增强检查、.3真实32窗口续训均COMPLETED、ExitCode0:0，分别约0秒/4秒/8秒/11秒。Linux日志WALLTIME_SIGNAL_PASSED确认SIGUSR1后当前轮结束并保存；REAL_RESUME_PASSED确认真实GTENet/脑电1→2轮及正式源epoch100元数据。
- 正式训练步骤.4为RUNNING；主作业观察Elapsed00:00:57。所有计算在已分配c002资源内完成，未绕过调度。
- 已将当前stdout和source.sha256读取到本地 runs/deap_window_split_adaptive/1887873/，仅为读取时快照。新实验仍未结束，不把程序启动或前置检查成功写成2000轮训练完成。

### 1887873：训练完成补记

记录日期：2026-10-22。证据来自2026-10-20 20:27（北京时间）通过SSH实际读取的sacct、服务器stdout/stderr及产物目录；本次只同步文档，未重新预测、未启动训练或提交作业。本补记取代上文“正式第101轮已启动、尚无终态”的状态快照，保留历史记录。

- 调度终态：主作业1887873及正式训练步骤.4均COMPLETED、ExitCode=0:0，节点c002。主作业2026-10-19 19:00:42开始，2026-10-20 12:12:39结束，Elapsed=17:11:57。
- stdout明确输出 `STOP_REASON=early_stopping; completed_epochs=378`、`TRAINING_EXIT_REASON=early_stopping`、`Training finished.`、`ADAPTIVE_ATTEMPT_FINISHED`。第338轮更新早停参考值，之后40轮没有达到累计0.001的改善阈值，第378轮计数达到40/40；属于约定规则下完成，不是跑满2000轮，也不是超时或报错。
- 数据、划分、参数、资源及代码沿用本实验提交记录和协议第11节：DEAP Valence <5/≥5二分类，32名受试者、76,800个1秒窗口，窗口级random_split seed42，61,440训练/15,360验证，n_slices=1；从第100轮last状态执行第101–378轮，共278个新epoch。全局随机性未统一固定。
- 完整提交命令：`cd /share/home/ycg_luanjingjie/GTE-Net && sbatch --nodelist=c002 runs/adaptive_code_20261019_v1/gte_adaptive_cpu.slurm /share/home/ycg_luanjingjie/GTE-Net/runs/adaptive_code_20261019_v1`。
- 运行版本为HEAD `a6facc96b2c159abe14b8d88faf95564d2634122` 的dirty working tree派生冻结副本；训练入口及控制模块SHA256沿用本实验提交记录，完整清单为输出目录code.sha256。本次文档同步前本地HEAD仍相同，既有未提交修改保留，未提交或推送。

| 位置 | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 第338轮，最佳轮，日志四位小数 | 0.5128 | 0.7797 | 0.5569 | 0.7549 |
| 第378轮，最后一轮，日志四位小数 | 0.5121 | 0.7813 | 0.5552 | 0.7528 |

- 最佳模型保存日志和文件名包含完整Accuracy：Train=0.7797037760416666、Val=0.7548828125（75.48828125%）。这是训练程序报告的整集合Accuracy；本次没有通过独立重载重新计算。
- 已实际列目录确认以下服务器产物存在且非空，根目录为 `/share/home/ycg_luanjingjie/GTE-Net/runs/deap_window_split_adaptive/1887873/`：最佳模型 `GTE_best_train_acc_0.7797037760416666_val_acc_0.7548828125.pth`（约3.3M）、`GTE_best_last.pth`（约3.3M）、同最佳模型前缀的 `_final_curves.png`（约100K）；本次未下载最终产物或视觉审阅曲线。
- 日志：服务器 `logs/slurm-1887873.out`、`logs/slurm-1887873.err`；stderr尾部是正常验证进度条。结束日志中旧源检查点校验为OK。
- 结果边界：同一窗口级验证集上的单次续训分支，非独立重复、非受试者级测试；无多次均值/标准差，无Macro-F1、混淆矩阵或逐受试者成绩。上游约80.44%仍是上游报告值，尚未复现。
- 用户2026-10-22明确表示“不需要核实”，因此不执行最佳模型独立重载复核；该项保持“未执行”，不写为PASSED，不作为本次窗口级训练任务完成门槛。本次训练任务已完成，整个项目的受试者级独立评估等目标仍未完成；后续实验不自动启动。

### 2026-11-06 发布前语法检查

- 环境：Windows，本地 `.venv`，发布前工作树。
- 实际命令：`.\.venv\Scripts\python.exe -m py_compile DatasetProcessSave.py GTE_train.py model_optimized.py validate_setup.py`
- 结果：退出码 0，四个文件语法检查通过。
- 本次仅准备 GitHub 发布，未运行训练、前向检查或重新评估；既有性能结论不变。

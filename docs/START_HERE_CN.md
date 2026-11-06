# GTE-Net 从这里开始

学院服务器的排错顺序作业：在远程项目根目录执行
`sbatch jobs/gte_repair_smoke_cpu.slurm`，先用 320 个真实窗口完成训练、验证、
检查点重载与曲线检查，通过后才运行全量 1 epoch。该脚本使用学院服务器绝对路径，
不用于 Windows。完整配置见 `docs/REPRODUCTION_PROTOCOL.md`。

本文中的命令均在项目根目录 `D:\workspace\GTE-Net` 执行，不在 `docs/` 中执行。

## 现在有什么

- 上游 GTE-Net 代码已经放入项目根目录。
- DEAP Python 预处理数据已经放在 `dataset/DEAP/data_preprocessed_python/`，共 32 名受试者。
- 第一阶段只做 DEAP 的 Valence 二分类，不使用 SEED、SEED-IV、SEED-VIG、原始 BDF 或面部视频。

## 第一步：建立环境并检查

在 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe validate_setup.py
```

检查脚本只读取 `s01.dat`，对一个真实 EEG 窗口做基线校正、DE 特征计算和模型前向计算；不会生成预处理缓存，也不会开始训练。

## 第二步：跑一个最小实验

完整数据首次运行需要计算并缓存 1 秒窗口和五频带 DE 特征。当前 CPU 实测一段试验约 2.26 秒，按 32 人、每人 40 段粗略估算，首次全量预处理约 0.8 小时；实际时间会受磁盘和系统负载影响。先使用 1 个 epoch 验证整条链路：

```powershell
.\.venv\Scripts\python.exe GTE_train.py `
  --dataset DEAP `
  --data_path .\dataset\DEAP\data_preprocessed_python `
  --preprocessed_data_path .\dataset\DEAP\data_time_window_1s `
  --epochs 1 `
  --batch_size 32 `
  --time_window 1 `
  --model_save_path .\runs\smoke\GTE_best.pth
```

首次预处理成功后，再把 `--epochs` 调到 50，观察损失和验证准确率是否正常，再决定是否做 700 轮。

## 第三步：把“跑通”升级为可信实验

上游训练脚本用窗口级随机切分。同一名受试者及相邻时间窗口可能同时出现在训练集和验证集，因此其验证准确率只能算代码复现指标，不能当作跨受试者泛化结果。

正式实验应增加按受试者切分：例如 32 人中按受试者划分训练、验证、测试，或做 leave-one-subject-out。还要固定随机种子、保存每次配置，并报告准确率、宏平均 F1、混淆矩阵和各折结果。完成这些改造后，再与 README 中约 80.4% 的窗口级结果比较。

## 预期产出

1. 环境与数据检查通过。
2. 1 epoch 全链路运行成功，生成缓存、检查点和训练曲线。
3. 复现上游窗口级结果，明确它的验证边界。
4. 改成受试者级评估，得到可用于报告的结果。

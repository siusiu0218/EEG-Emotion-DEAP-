# AGENTS.md

## 项目目标

本仓库用于学习和复现上游项目 [Yuzehua00/GTE-Net](https://github.com/Yuzehua00/GTE-Net)。当前第一阶段目标是：

1. 使用 DEAP Python 预处理数据复现 Valence 二分类；
2. 先验证上游窗口级随机划分流程；
3. 再实现受试者级独立评估，得到更可信的泛化结果。

这是一份实验性 PyTorch 研究代码，不是临床工具。

## 完成标准

一次修改只有同时满足以下条件才算完成：

- 请求的代码或文档已经交付；
- 运行了与风险相称的检查；
- `docs/EXPERIMENT_LOG.md` 中只记录真实运行过的实验；
- 已验证事实、推断、上游报告值和待验证事项明确区分；
- 如改变数据定义、划分、标签、指标或训练入口，同步更新相关指导文件。

## 当前权威入口

以下路径均相对项目根目录。除 `README.md`、`AGENTS.md`、`CONTEXT.md` 外，Markdown 文档集中在 `docs/`；教学技能原先约定的根目录 Markdown 文件也使用该实际位置。

- 项目目标、当前状态与下一步：`CONTEXT.md`
- 总览与运行命令：`README.md`
- 最短上手路线：`docs/START_HERE_CN.md`
- 数据目录与用途：`docs/DATA_LAYOUT_CN.md`
- 实验定义：`docs/REPRODUCTION_PROTOCOL.md`
- 实际运行记录：`docs/EXPERIMENT_LOG.md`
- 相对上游修改：`docs/UPSTREAM_CHANGES.md`
- 主训练入口：`GTE_train.py`
- 快速验证入口：`validate_setup.py`
- 教学目标、路线与接续：`docs/MISSION.md`、`docs/LEARNING_PLAN.md`、`docs/LEARNING_PROGRESS.md`

用户要求学习、做题或继续课程时，先读取以上教学文件及 `docs/NOTES.md`，再核对相关代码。每次只推进一个小任务，用户先做，助手逐级提示与评审；有实际学习证据后更新进度。仅讲过或助手代做不算用户掌握。实验状态仍以 `docs/EXPERIMENT_LOG.md` 为准。该教学请求本身不授权助手代跑训练或修改研究代码。

若文档与代码冲突，以当前代码行为和可重复验证结果为准；随后修正文档，不要保留冲突描述。

## 范围与边界

- 当前标准路线只使用 `dataset/DEAP/data_preprocessed_python/s01.dat` 至 `s32.dat`。
- 第一阶段只做 DEAP Valence 二分类：Valence `< 5` 为 0，`>= 5` 为 1。
- 不要把 DEAP、SEED、SEED-IV 或 SEED-VIG 混成一个数据集。
- `SEEDDataset` 仍含假设性解析；未修复并验证前，不得称 SEED 路线可复现。
- 当前 `random_split` 是窗口级划分，存在同一受试者和相邻窗口跨集合的风险。
- 上游约 80.44% 是上游报告值，不是本地已复现结果，也不是跨受试者性能。
- 当前主训练入口使用模型默认 `n_slices=1`；不要宣称已经验证多窗口时序建模。
- 不要将数据文件、缓存、检查点或训练曲线提交到 Git。

## 数据保护

- `dataset/` 和 `datas/` 是用户提供的数据资产；默认只读。
- 不移动、删除、覆盖或重新打包原始下载文件，除非用户明确要求。
- 预处理缓存只能写入 `dataset/DEAP/data_time_window_1s/` 或用户明确指定的输出目录。
- 删除缓存或运行产物前必须确认精确目标；优先采用可恢复方式。

## 标准命令

在 Windows PowerShell、项目根目录下运行：

```powershell
.\.venv\Scripts\python.exe validate_setup.py
```

快速语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  DatasetProcessSave.py `
  GTE_train.py `
  model_optimized.py `
  validate_setup.py
```

1 epoch 冒烟实验使用 `README.md` 和 `docs/REPRODUCTION_PROTOCOL.md` 中的冻结命令。除非修改本身涉及训练链路，否则不要擅自启动耗时的完整训练。

## 修改原则

- 保留用户的未提交改动，不回退或覆盖无关文件。
- 优先做最小、清晰、可验证的修改。
- 修复复现问题时，先建立能暴露问题的最小检查，再修改代码。
- 改动预处理时，至少核对受试者数量、原始数组形状、窗口数量、首末窗口位置、特征形状和有限值。
- 改动模型时，至少运行真实 DEAP 窗口的前向检查；不要只用随机张量证明正确。
- 改动训练指标时，使用完整预测与标签验证计算，避免平均“逐批累计准确率”等偏差。
- 不因脚本能运行就声称研究结论有效；数据划分、统计单位和指标定义也必须成立。

## 文档同步规则

| 变更类型 | 必须同步的文件 |
| --- | --- |
| 项目目标、阶段、关键决策或下一步 | `CONTEXT.md` |
| 数据目录、格式或数据用途 | `docs/DATA_LAYOUT_CN.md`、`README.md` |
| 标签、窗口、划分、指标或超参数 | `docs/REPRODUCTION_PROTOCOL.md`、`README.md` |
| 相对上游的代码或配置变化 | `docs/UPSTREAM_CHANGES.md` |
| 实际执行的验证或训练 | `docs/EXPERIMENT_LOG.md` |
| 最短运行命令改变 | `docs/START_HERE_CN.md`、`README.md` |

`docs/EXPERIMENT_LOG.md` 是实验记录，不得把计划写成已完成，也不要覆盖历史条目；需要纠正时新增更正说明。

## 结果表述

结果必须注明：

- 数据集与任务；
- 数据划分单位；
- 样本数量或受试者数量；
- 完整运行命令或配置；
- 代码提交或工作树状态；
- Accuracy、Macro-F1 等指标及其计算范围；
- 单次结果还是多次运行的均值与标准差。

推荐用语：

- “环境与前向检查通过”；
- “窗口级 80/20 随机划分验证准确率为……”；
- “受试者级独立评估结果为……”。

禁止仅写“模型准确率为 80%”而省略划分与验证边界。


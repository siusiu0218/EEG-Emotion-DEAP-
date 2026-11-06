# S1-E01：实现单受试者数据读取模块

这是新教学形式下的第一道项目编程题。后续切窗模块将使用本题返回的 EEG 和每次试验的类别标签。

## 任务与接口

补全同目录 `read_subject.py` 中的 `load_subject(file_path)`。

- 输入：官方 DEAP Python 预处理受试者文件路径；默认运行 `s01.dat`。
- 文件读取已提供，得到字典 `subject`。
- `subject['data']` 形状 `(40,40,8064)`：试验、通道、采样点；前 32 个通道是本题使用的 EEG。
- `subject['labels']` 形状 `(40,4)`：每次试验的 Valence、Arousal、Dominance、Liking；Valence 是第 0 列。
- 返回 `eeg, targets`：EEG 形状 `(40,32,8064)`；类别标签为整数 NumPy 数组，形状 `(40,)`，Valence < 5 为 0、>= 5 为 1。

保留全部试验和采样点，保持 EEG 与标签顺序一致。允许循环或 NumPy 操作，不要求最短代码。不修改原数据，不启动训练。本题暂不切窗、不去基线。

## 必要工具

- NumPy 多维索引中，逗号分隔各维；单独 `:` 保留这一维全部元素，`:n` 取索引 0 到 n-1；整数索引选择一个位置并去掉该维。
- 数组与数值比较可得到布尔数组；需要时使用 `.astype(np.int64)` 转换整数类型。也可以逐项构造标签。
- `return a, b` 让函数返回两项；调用端用 `a, b = ...` 接收。
- 不熟悉其中语法时，请结合当前 TODO 提问；无需先回答独立理论题。

## 运行与验收

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe exercises/S1-E01/read_subject.py
```

应输出 EEG shape `(40,32,8064)`、Targets shape `(40,)`、整数标签类型和前 10 个标签。前 10 个标签的具体值由数据决定，不预填答案。

脚本中的断言只检查基本形状、类型与取值，不能证明通道内容、顺序和阈值映射正确。提交函数代码与完整输出；助手据此核对正确通道、Valence 列、顺序与边界（特别是等于 5）。提交代码但未运行时明确说明。

## 依据与状态

依据：当前 `DatasetProcessSave.py:177-189` 和 `docs/REPRODUCTION_PROTOCOL.md` 第 3、4 节。讲义见 `lessons/0002-load-subject-exercise.html`。

已提供读取与运行骨架，核心实现留给用户；尚无作答或运行证据。助手未代跑此练习。

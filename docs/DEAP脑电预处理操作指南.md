# DEAP 脑电预处理操作指南

本指南按照我们对话中的顺序，带你从一个人的一次试验开始，得到可以交给 GTE-Net 的一秒脑电样本，再理解如何处理全部数据、保存样本和划分训练集与验证集。第十二步把模型内部的处理和训练过程拆成小环节，逐步解释。

使用 DEAP Python 预处理数据。它已经经过数据提供方的预处理；这里做的是项目需要的进一步切窗、基线校正和特征计算，不是从原始 BDF 重建全部信号处理流程。

## 第一步 找到一个人的数据文件

在项目根目录 `D:\workspace\GTE-Net` 下，找到：

```text
dataset/DEAP/data_preprocessed_python/s01.dat
```

`s01.dat` 对应一名受试者，其中有 40 次试验。先用这个文件学习，后续再对 `s01.dat` 到 `s32.dat` 全部处理。

需要的技术：文件路径、`pickle`、NumPy。

```python
from pathlib import Path
import pickle
import numpy as np

data_path = Path("dataset/DEAP/data_preprocessed_python/s01.dat")
with data_path.open("rb") as handle:
    subject = pickle.load(handle, encoding="latin1")

data = np.asarray(subject["data"])
labels = np.asarray(subject["labels"])

print(data.shape)    # 预期 (40, 40, 8064)
print(labels.shape)  # 预期 (40, 4)
```

只读取你已经取得的官方数据文件。`encoding="latin1"` 用于读取这一旧格式数据。

数组维度的含义：

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| `data` | `(40, 40, 8064)` | 40 次试验、40 个通道、每个通道 8064 个采样点 |
| `labels` | `(40, 4)` | 每次试验对应四个评分 |

四个评分依次为 Valence、Arousal、Dominance、Liking。当前任务只取 Valence。

## 第二步 取出一次试验的 EEG

先选第一名受试者的第一次试验，取前 32 个 EEG 通道：

```python
trial_index = 0
eeg = data[trial_index, :32, :]
valence = float(labels[trial_index, 0])
label = int(valence >= 5.0)

print(eeg.shape)  # 预期 (32, 8064)
print("Valence:", valence, "二分类标签:", label)
```

Python 索引从 0 开始，所以 `trial_index = 0` 是第一次试验。`:32` 取编号 0 到 31 的通道。

需要的技术：数组索引、切片、标签阈值。

完成后，你拿到一整次试验的 32 通道脑电和该试验的标签。

## 第三步 分清基线和看视频的脑电

采样率是 128 Hz，也就是每个通道每秒有 128 个采样点。

```text
一次试验共 63 秒
├── 前 3 秒：基线，3 × 128 = 384 个点
└── 后 60 秒：刺激期，60 × 128 = 7680 个点
```

```python
baseline = eeg[:, :384]
stimulus = eeg[:, 384:8064]

print(baseline.shape)  # 预期 (32, 384)
print(stimulus.shape)  # 预期 (32, 7680)
```

这里的“看视频的脑电”指刺激期信号。当前流程直接使用 EEG 数据，不需要打开面部视频文件。

需要的技术：采样率与时间换算、切片边界。

## 第四步 从刺激期取出一秒

学习时先取视频开始后的第一秒，方便重复核对。正式预处理则使用全部 60 个一秒窗口，不是随便挑一秒。

从整次试验 `eeg` 中取：

```python
window_1 = eeg[:, 384:512]
window_2 = eeg[:, 512:640]
window_60 = eeg[:, 7936:8064]
```

如果从已经截好的 `stimulus` 中取，索引要从 0 开始：

```python
window = stimulus[:, 0:128]
```

这和 `eeg[:, 384:512]` 是同一段数据。不要从 `stimulus` 中再跳过 384 个点，否则会错过前 3 秒刺激期。

切片 `384:512` 包含 384，不包含 512，恰好取出 128 个点。各窗口互不重叠。

需要的技术：切片的左闭右开规则、窗口长度和步长。

完成标准：一个窗口的形状为 `(32, 128)`。

## 第五步 把三秒基线变成一秒平均基线

三秒基线有 384 个点，一秒窗口有 128 个点，不能直接相减。先把基线拆成三个一秒片段，再按对应位置平均：

```python
baseline_1 = eeg[:, 0:128]
baseline_2 = eeg[:, 128:256]
baseline_3 = eeg[:, 256:384]

baseline_mean = (baseline_1 + baseline_2 + baseline_3) / 3
print(baseline_mean.shape)  # 预期 (32, 128)
```

例如某通道三个片段的第一个点分别是 6、7、8，那么平均基线的第一个点是 7。每个通道、每个对应位置都这样计算。

注意：这里不是把每个通道的 384 个点平均成一个数，而是得到每个通道的一段 128 点基线。

需要的技术：对应位置求平均、数组逐元素运算。

## 第六步 做基线校正

```python
window_corrected = window - baseline_mean
print(window_corrected.shape)  # 预期 (32, 128)
assert np.isfinite(window_corrected).all()
```

同一通道、对应位置相减。如果刺激期一个点是 10，平均基线对应的点是 7，校正后就是 3。

这样做是减去观看视频前的背景信号。它不保证剩下的信号全部由情绪引起，也不等于消除了所有噪声。

校正后的 `(32, 128)` 数组就是当前样本的时域输入，需要保留。

## 第七步 计算五个频带的 DE 特征

对校正后的每个通道，分别通过五个带通滤波器，得到不同频率范围的信号。

| 频带 | 频率范围 |
| --- | --- |
| δ | 0.5–4 Hz |
| θ | 4–8 Hz |
| α | 8–13 Hz |
| β | 13–30 Hz |
| γ | 30–50 Hz |

对每段滤波后的信号计算方差，再按高斯分布的差分熵公式计算 DE：

```python
de = 0.5 * np.log(2 * np.pi * np.e * variance)
```

方差描述数值波动大小。DE 在这里是根据滤波后信号的方差构造的特征，不能直接解释为某种情绪。

当前项目的 `calculate_de` 使用 8 阶 Butterworth 带通滤波和 `scipy.signal.filtfilt`，每个通道输出五个值。学习时直接调用该函数：

```python
from DatasetProcessSave import calculate_de

freq_features = np.stack([
    calculate_de(channel, dataset_type="DEAP")
    for channel in window_corrected
])

print(freq_features.shape)  # 预期 (32, 5)
assert np.isfinite(freq_features).all()
```

需要的技术：带通滤波、方差、对数、差分熵、循环、`np.stack`。

`calculate_de` 会对非正方差和非有限 DE 值做保护；输出有限只说明通过这个检查，不代表滤波过程和研究结论已经充分验证。仓库中的快速 DE 函数使用不同滤波过程，先不要替换。

## 第八步 给这一秒样本配上标签

当前 Valence 二分类规则是：

```python
label = int(valence >= 5.0)
```

Valence 小于 5 为 0，大于等于 5 为 1。一次试验产生的 60 个窗口都沿用该试验的同一个评分标签；数据没有给每一秒单独打分。

现在你得到一个完整样本：

| 内容 | 形状或取值 | 用途 |
| --- | --- | --- |
| `window_corrected` | `(32, 128)` | 时域输入 |
| `freq_features` | `(32, 5)` | 五频带 DE 输入 |
| `label` | 0 或 1 | 训练目标 |

## 第九步 把一次试验的60秒全部处理完

前八步只处理了第一次试验的第一秒。接下来仍然使用这一次试验，把刺激期的第1秒到第60秒依次处理，每秒生成一个样本。

每个窗口都要做两件事：减去这次试验的平均基线，再计算五频带 DE。标签沿用这次试验的标签。

下面代码接在前八步之后执行，使用已经得到的 `eeg`、`baseline_mean` 和 `label`：

```python
trial_samples_time = []
trial_samples_freq = []
trial_labels = []

for second_index in range(60):
    start = 384 + second_index * 128
    end = start + 128
    window_corrected = eeg[:, start:end] - baseline_mean
    freq_features = np.stack([
        calculate_de(channel, dataset_type="DEAP")
        for channel in window_corrected
    ])

    trial_samples_time.append(window_corrected)
    trial_samples_freq.append(freq_features)
    trial_labels.append(label)

trial_samples_time = np.stack(trial_samples_time)
trial_samples_freq = np.stack(trial_samples_freq)
trial_labels = np.asarray(trial_labels, dtype=np.int64)

assert trial_samples_time.shape == (60, 32, 128)
assert trial_samples_freq.shape == (60, 32, 5)
assert trial_labels.shape == (60,)
assert np.isfinite(trial_samples_time).all()
assert np.isfinite(trial_samples_freq).all()
assert (trial_labels == label).all()
```

`second_index` 从0到59。第一次取 `384:512`，最后一次取 `7936:8064`，恰好覆盖60秒刺激期，不包含基线，不重叠。

| 内容 | 形状 | 含义 |
| --- | --- | --- |
| `trial_samples_time` | `(60, 32, 128)` | 60个窗口，每个窗口有32通道、128个采样点 |
| `trial_samples_freq` | `(60, 32, 5)` | 60个窗口，每个窗口有32通道、每通道5个DE值 |
| `trial_labels` | `(60,)` | 60个窗口各有一个标签，这次试验内全部相同 |

需要的技术：`for`循环、列表、`append`、`np.stack`、形状检查。

完成标准：一次试验得到60个样本，时域输入、DE输入和标签的顺序一一对应。

## 第十步 处理全部试验和受试者，再保存样本

第九步完成的是“一个人的一次试验”。下一步有两层重复：

1. 在同一个人的文件中，依次处理40次试验；每次都重新取该试验的基线和Valence评分。
2. 再依次读取 `s01.dat` 到 `s32.dat`，对每个人重复上述操作。

不能把第一次试验的平均基线或标签拿给其他试验使用。每次试验有自己的基线和标签。

```text
一个窗口 → 一个样本
一次试验 → 60个样本
一个人   → 40 × 60 = 2,400个样本
全部32人 → 32 × 40 × 60 = 76,800个样本
```

这是32个文件均符合预期结构时应得到的数量，不表示已经完成了全量处理。

为了避免每次训练都重新切窗和计算DE，当前项目按受试者保存为 `.npz` 文件：

```text
dataset/DEAP/data_time_window_1s/
├── subject_01.npz
├── subject_02.npz
├── ...
└── subject_32.npz
```

一个受试者缓存包含：

| 保存字段 | 预期形状 | 内容 |
| --- | --- | --- |
| `samples_time` | `(2400, 32, 128)` | 这个人的全部时域窗口 |
| `samples_freq` | `(2400, 32, 5)` | 与窗口对应的五频带DE特征 |
| `labels` | `(2400,)` | 与窗口对应的二分类标签 |

可以把 `.npz` 理解为一个装着多个NumPy数组的文件。时域、DE和标签在相同位置上属于同一个样本，保存和读取时不能把它们的顺序弄乱。

当前项目的 `DatasetProcessSave.py` 中，`save_preprocessed_data` 负责全量处理与保存，`load_preprocessed_data` 负责读取缓存。标准训练入口 `GTE_train.py` 在缓存缺失时会调用预处理，不需要先手写另一套全量处理脚本。实际运行命令见 `docs/REPRODUCTION_PROTOCOL.md` 第7.3节；第一次全量处理可能较慢，学习到这里不需要立即启动训练。

缓存是新增的处理结果，原来的 `s01.dat` 到 `s32.dat` 保持不变。若改变窗口长度、基线方法或频带定义，应使用新的缓存目录，避免混用不同方法生成的数据。

需要的技术：嵌套循环、批量文件读取、`np.savez`、`np.load`、样本顺序与数量核对。

完成标准：全量处理后应有32个受试者缓存，每人2,400个窗口，总计76,800个窗口；还需核对形状、标签对应关系和有限值，不能仅凭文件存在就认定处理正确。

## 第十一步 划分训练集和验证集

保存好样本后，需要先分出两部分：

- **训练集**：模型用这些样本的输入和标签学习，调整自己的参数。
- **验证集**：用来检查模型的预测与真实标签是否一致，不用这些样本的损失更新模型参数；当前训练流程会根据验证结果选择检查点和调整学习率。

两部分都包含时域输入、DE输入和标签。划分的是完整样本，不能把一秒的脑电放进训练集，却把对应的DE或标签放进验证集。

当前 `GTE_train.py` 的上游复现路线是：将全部窗口合并，使用 `random_split` 按80%和20%随机划分，划分种子为42。

| 集合 | 比例 | 全量数据下的窗口数量 |
| --- | --- | --- |
| 训练集 | 80% | 61,440 |
| 验证集 | 20% | 15,360 |

这是按“窗口”划分，不是按“人”划分。例如，同一个人同一次试验的第1秒可能进入训练集，第2秒可能进入验证集。这些样本有相似背景，存在信息泄漏风险，因此窗口级验证准确率不能当成模型对新受试者的识别能力。

项目先用窗口级划分核对上游代码流程；后续再实现受试者级独立评估，确保同一个人不会跨训练、验证或测试集合。当前受试者级评估尚未实现，当前窗口级流程也没有独立测试集。

需要的技术：训练集与验证集、样本索引、随机划分、随机种子、信息泄漏。

完成标准：能说明哪些样本参与学习、哪些样本用于验证，并清楚当前划分单位是窗口。第十一步结束时，数据已经分好；接下来在第十二步解释怎样交给模型训练。

## 第十二步 从一个样本到模型训练，一点点理解

这一步包含一整套模型处理和训练流程，不是一个简单操作。先从一个样本开始理解，再扩展到一批样本。以下结构和维度对应当前 `model_optimized.py`、`GTE_train.py` 及冻结的 DEAP 配置；示意数字只用于解释，不是实验结果。

阅读顺序：先读大白话和例子，再看公式，最后对照代码。不需要一开始就背公式。下面代码标为“项目摘录”的是当前代码中的核心语句；省略了无关部分和部分原注释，不能当作独立脚本直接运行。“教学示意”用于说明数学关系，不是替换项目实现。

统一记号：`B`是批次样本数，`M`是EEG通道数，`T`是采样点数，`D`是某一步的特征维度，`S`是序列位置数。当前示例中 `B=32`、`M=32`、`T=128`、`S=1`，但不同环节的 `D` 会变化。公式中的求和符号 `Σ` 表示把指定项加起来；矩阵乘法表示一组加权求和；`[a || b]` 表示拼接。

### 12.1 先分清输入材料和正确答案

一个样本有三样东西：

| 内容 | 形状或取值 | 可以怎样理解 |
| --- | --- | --- |
| 校正后的一秒脑电 | `(32, 128)` | 32行、128列的数字表，每行一个通道，每列一个采样时刻 |
| 五频带DE | `(32, 5)` | 32行、5列的数字表，每行是一个通道的五频带DE |
| 标签 | `0`或`1` | 这个样本对应的正确类别 |

脑电和DE是交给模型看的材料，标签是训练时核对的答案。模型前向计算只接收脑电和DE，不把标签作为输入。

为什么给两份材料？脑电波形保留了数值随时间怎样变化；DE把各频带的波动情况概括成了特征。它们来自同一秒脑电，描述的角度不同。

**对应代码与记号**

```python
# 项目摘录：GTE_train.py，DEAPDataset.__getitem__
x_time = torch.FloatTensor(sample['time'])
x_freq = torch.FloatTensor(sample['freq'])
label = torch.LongTensor([label]).squeeze()
return x_time, x_freq, label
```

`FloatTensor`用于浮点输入；`LongTensor`用于整数类别索引；`squeeze()`把单个标签的长度1维度去掉，使它成为标量。一个样本可记为 `(X, F, y)`，分别对应脑电、DE和标签；组批后标签是长度为 `B` 的整数Tensor。

### 12.2 转成Tensor，把样本叠成一批

当前数据集类把NumPy数组转换为PyTorch的Tensor（张量）。现在可以把Tensor理解成模型能够处理的数字数组；转换本身不等于提取特征。

每个脑电样本是一张二维数字表，不需要先拉成一条向量。把32个样本叠起来，可以想象成32张纸摞在一起，每张纸都有32行、128列：

```text
一个脑电样本：(32, 128)
32个样本叠起：(32, 32, 128)
               ↑   ↑    ↑
             样本数 通道数 采样点数

一组DE：      (32, 5)
32组DE叠起：  (32, 32, 5)
32个标签：    (32,)
```

叠起来不会把这些样本平均掉，也不会拼成一个32秒长的样本。每个样本仍占一个独立位置。

例如，`batch_time[0, 2, 10]` 表示第1个样本、第3个通道、第11个采样点的数值，索引从0开始。

下面是NumPy层面的示意，要求 `windows` 已经包含32个形状为 `(32, 128)` 的数组：

```python
batch_time = np.stack(windows, axis=0)
# batch_time.shape == (32, 32, 128)
```

`axis=0` 是增加最外面的样本维度。实际训练中，PyTorch的 `DataLoader` 会把数据集返回的Tensor自动组成批次。第几个脑电样本，就必须配第几组DE和第几个标签，不能错位。

**一批只表示一起送入模型，不表示同一个时间点或连续的时间段。** 训练加载器启用了打乱顺序，一批可能包括不同人、不同试验、不同秒的窗口，例如：

```text
样本1：受试者A，第1次试验，第8秒
样本2：受试者B，第3次试验，第25秒
样本3：受试者A，第2次试验，第10秒
```

这32个样本不是同一条时间序列的32个位置。后面Transformer的序列注意力不会把它们当作前后几秒来比较。训练时卷积中的BatchNorm会使用批次统计，但这也不等于把批次组织成时间序列。

**对应代码与形状**

```python
# 项目摘录：GTE_train.py，train
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# 教学示意：PyTorch把多个样本沿最外层叠起来
# batch_time = torch.stack(list_of_time_tensors, dim=0)
```

组批的数学意思是 `X_batch[b, m, t] = X_b[m, t]`：第 `b` 个样本的第 `m` 通道、第 `t` 点，原样放入批次对应位置。`shuffle=True`打乱取样顺序，不改变同一个样本内部的通道、采样点和配对关系。

### 12.3 脑电波形进入一维卷积

先只看一个样本里的一个通道：它有128个数，按时间排列。一维卷积用一个小窗口沿时间移动，这个小窗口叫卷积核。

假设卷积核长度为3，只是为了讲原理：

```text
第一次看：[第1点，第2点，第3点]
第二次看：       [第2点，第3点，第4点]
第三次看：              [第3点，第4点，第5点]
```

每读取一段，就计算一个新特征值：

```text
新特征值 = 第一个点×权重1 + 第二个点×权重2
         + 第三个点×权重3 + 偏置
```

例如输入 `[2, 5, 3]`，示意权重 `[-1, 0, 1]`，偏置为0，结果是 `2×(-1)+5×0+3×1=1`。这组权重比较这段的末端与开端；换一组权重，就会对另一种变化敏感。实际权重通过训练学习，不能提前说某个卷积核识别某种情绪。

卷积核一路移动，产生一条特征序列。多个卷积核可以提取不同模式。当前代码使用长度为15、31、63的三路卷积，每路输出32条特征序列，再拼接，通过长度为1的卷积组合成64条特征序列。不同长度读取的局部时间范围不同。中间还使用BatchNorm和GELU激活函数。

同一套卷积参数会用于各EEG通道，之后才由GAT结合通道信息。一批输入 `(32, 32, 128)` 的卷积输出为 `(32, 32, 64, 128)`：样本、EEG通道、特征序列、时间位置。

前面把60秒切成一秒，是在制作样本；这里卷积核在一秒样本内移动，是在提取特征，二者不是同一种切窗。

**卷积公式：把小窗口里的数加权相加**

对单个输入通道、第 `r` 个卷积输出特征，用当前步长1、膨胀率1的情况表示：

$$
u_{r,t}=b_r+\sum_{i=0}^{k-1}w_{r,i}\,\widetilde{x}_{t+i-p}
$$

`k`是卷积核长度；`p=k//2`是两端填充长度；`w`是可学习权重；`b_r`是偏置；`t`是输出时间位置。`x̃`表示越过原始边界的位置填0，因此两端也能计算。PyTorch的Conv1d执行的是通常称为卷积的互相关计算，不在这里翻转核。

输出长度为：

$$
T_{out}=\left\lfloor\frac{T+2p-d(k-1)-1}{s}+1\right\rfloor
$$

`s`是步长，`d`是膨胀率，向下取整记为 `⌊ ⌋`。当前 `s=d=1`，核长为奇数、`p=k//2`，所以输出仍有128个时间位置。

```python
# 项目摘录：model_optimized.py，Conv1DBankShared
nn.Conv1d(1, bank_out_channels, k, stride=1, padding=(k // 2))
nn.BatchNorm1d(bank_out_channels)
nn.GELU()

# forward中的核心语句
B, M, T = x.shape
x_r = x.view(B * M, 1, T)
outs = [conv(x_r) for conv in self.convs]
cat = torch.cat(outs, dim=1)
pw = self.pointwise(cat)
pw = self.bn_pw(pw)
pw = self.act(pw)
pw = pw.view(B, M, pw.shape[1], T)
```

这里 `Conv1d(1, ...)` 的输入通道是1，因为代码先把每个EEG通道单独展开：`(B,M,T) → (B×M,1,T)`。这不是只保留一个EEG通道，而是让全部通道分别使用同一套卷积。三路各32条特征，拼成96条，再由 `self.pointwise`（`Conv1d(96,64,1)`）在每个时间位置组合成64条。

BatchNorm在训练时对每个卷积特征通道，使用展开批次和时间位置上的统计量。其基本计算是 `γ(x−μ)/sqrt(σ²+ε)+β`；`μ`和`σ²`为统计量，`γ`和`β`可学习，`ε`防止除零。GELU为非线性激活，定义为 `GELU(x)=xΦ(x)`，`Φ`是标准正态分布的累积分布函数。现在只需理解：它不只是再做一次线性加权。

### 12.4 DE进入全连接层

DE走另一条路。一个通道原本有5个DE值，全连接层的一个输出会把全部5个值分别乘权重，再相加：

```text
一个新特征 = δ的DE×权重1 + θ的DE×权重2
           + α的DE×权重3 + β的DE×权重4
           + γ的DE×权重5 + 偏置
```

因为一个输出与全部5个输入相连，所以叫全连接。用多组权重，就能产生多个输出。当前代码用全连接层和GELU把每通道5个DE值转换为64个特征值，一批的形状从 `(32, 32, 5)` 变成 `(32, 32, 64)`。

这没有增加新的测量信息，而是在学习五频带特征怎样组合。卷积沿时间读取局部波形；这里的全连接层一次看齐一个通道的5个DE值。

**全连接公式与代码**

$$
u_r=\sum_{j=1}^{5}W_{rj}f_j+b_r,\qquad v_r=\operatorname{GELU}(u_r)
$$

`f_j`是第 `j` 个频带DE，`r=1,…,64`表示输出的64个特征；每个输出有自己的一组权重和偏置。先线性组合，再激活。

```python
# 项目摘录：model_optimized.py，GTENet
self.freq_proj = nn.Sequential(nn.Linear(n_freq_bands, 64), nn.GELU())
freq_feat_proj = self.freq_proj(freq_features)
```

`n_freq_bands=5`。`nn.Linear(5,64)`作用于最后一维，所以 `(B,32,5) → (B,32,64)`，不会把不同样本或EEG通道揉成一个输入向量。

### 12.5 汇总波形特征，再与DE特征拼接

卷积产生的每条特征序列先沿时间求平均。例如示意序列 `[2, 4, 6, 8]` 汇总成一个数 `5`。64条序列分别平均，就得到每通道64个波形特征值。

随后，把同一个通道的两份特征首尾拼起来：

```text
波形特征：[a1, a2, ……, a64]
DE特征： [b1, b2, ……, b64]
                 ↓ 拼接
通道特征：[a1, ……, a64, b1, ……, b64]
```

拼接是首尾连接，不是相加或求平均。每通道得到128个特征，一个样本得到 `(32, 128)`，一批得到 `(32, 32, 128)`。

这里的128已经是“64个波形特征+64个DE特征”，不再是128个原始采样点。虽然形状与原始脑电相同，含义已经改变。

**平均和拼接的公式与代码**

$$
\bar{u}_{m,r}=\frac{1}{T}\sum_{t=1}^{T}u_{m,r,t},\qquad
h_m=[\bar{u}_m\;||\;v_m]
$$

`m`是EEG通道，`r`是卷积特征编号；对每条序列的 `T=128` 个位置求平均。`h_m`是该通道拼接后的128维特征。

```python
# 项目摘录：model_optimized.py，_spatial_pipeline
time_feat = self.conv_frontend(x_slice)             # (B,32,64,128)
time_feat_pooled = time_feat.mean(dim=-1)            # (B,32,64)
freq_feat_proj = self.freq_proj(freq_features)       # (B,32,64)
node_features = torch.cat([time_feat_pooled, freq_feat_proj], dim=-1)
# node_features: (B,32,128)
```

`dim=-1`指最后一维；`mean`是求平均，`cat`是沿已有维度连接。前面组批的 `stack` 则是增加一个新维度，两者用途不同。

### 12.6 GAT让不同通道交换信息

这一步叫图注意力网络（GAT，Graph Attention Network）。把32个EEG通道看作32个节点，每个节点带着自己的特征。

更新一个通道时，GAT计算应该从各通道取多少信息，再按权重汇总。例如只看三个通道，示意权重为：

```text
A自己的信息：0.5
B的信息：   0.3
C的信息：   0.2

A的新特征 = 0.5×A变换后的特征
          + 0.3×B变换后的特征
          + 0.2×C变换后的特征
```

每个通道都有自己的权重计算；换一个输入样本，权重也可能变化。当前训练调用没有提供邻接矩阵限制，GAT会计算所有通道对之间的注意力，包括通道自身。

处理后仍有32份通道特征，只是每份已经结合了其他通道的信息。注意力权重是模型计算结果，不能直接当作脑区真实生理连接的证据。

**单头GAT公式：先变换、打分、归一化，再汇总**

$$
g_i=Wh_i
$$

$$
e_{ij}=\operatorname{LeakyReLU}\left(a^\top[g_i\;||\;g_j]\right)
$$

$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{q=1}^{M}\exp(e_{iq})},\qquad
o_i=\sum_{j=1}^{M}\alpha_{ij}g_j
$$

`i`是正在更新的通道，`j`是提供信息的通道；`h_i`是输入特征，`W`把它投影为单头32维特征 `g_i`；`a`是可学习的打分向量，`aᵀ`表示转置。`e_ij`是通道对分数，`α_ij`是权重。固定 `i` 时，对所有 `j` 的权重之和为1。LeakyReLU对负数仍保留一个较小斜率，当前为0.2。

```python
# 项目摘录：model_optimized.py，GraphAttentionLayerDense.forward
Wh = self.W(h)
Wh_i = Wh.unsqueeze(2).expand(-1, -1, M, -1)
Wh_j = Wh.unsqueeze(1).expand(-1, M, -1, -1)
cat = torch.cat([Wh_i, Wh_j], dim=-1)
e = torch.matmul(cat, self.a).squeeze(-1)
e = self.leaky(e)
alpha = F.softmax(e, dim=-1)
out = torch.matmul(alpha, Wh)
```

`unsqueeze`增加维度，`expand`让每对通道都有一组待比较特征，`matmul`做矩阵乘法。单头第一层形状为：`h=(B,32,128)`、`Wh=(B,32,32)`、`e和alpha=(B,32,32)`、`out=(B,32,32)`。`alpha`后两个32分别表示接收与提供信息的通道，不是时间位置。

这段摘录省略了邻接掩码分支，因为当前主训练调用的 `adj=None`，没有限定哪些通道对可以交互。

### 12.7 多头和两层GAT分别是什么意思

“多头”是同时用几套独立的参数计算注意力，每套得到一份结果，再把结果拼接、加工。每个头关注什么由训练决定，没有提前规定某个头关注某个脑区或某种情绪。

“两层”是先更新一次通道特征，再用更新后的特征进行第二次信息汇总：

```text
合并后的通道特征 → 第一层GAT → 第二层GAT
```

当前冻结配置是2个头，每个头输出32个特征，因此每层输出的每通道特征维度为64。代码还有残差连接、Dropout，并在每层GAT之后使用LayerNorm。

残差连接把输入的一条变换路径加回输出，帮助保留信息；归一化调整数值尺度；Dropout在训练时随机将部分中间值置零。它们是模型处理的一部分，不代表已经证明提高了本地实验效果。

两层结束后，一批特征形状为 `(32, 32, 64)`，仍保留32个EEG通道。

**多头、残差和归一化的公式与代码**

将两个头的输出记为 `O¹`、`O²`，当前层的计算可概括为：

$$
C=[O^{1}\;||\;O^{2}],\qquad
Y=\operatorname{Dropout}(\operatorname{Linear}(C))+R(H)
$$

`H`是这层输入；`R`是残差路径：维度不同时用全连接层对齐，相同时直接保留。第一层从128维输入到64维输出，残差需投影；第二层输入输出都是64维，残差是恒等映射。

```python
# 项目摘录：model_optimized.py，MultiHeadGATLayer.forward
head_outs = [head(h, adj) for head in self.heads]
cat = torch.cat(head_outs, dim=-1)
cat = self.lin_out(cat)
cat = self.dropout(cat)
if self.residual:
    cat = cat + self.res_lin(h)

# _spatial_pipeline中，逐层调用
g1_out = self.gat1(node_features, adj)
g1_out_norm = self.ln1(g1_out)
g2_out = self.gat2(g1_out_norm, adj)
g2_out_norm = self.ln2(g2_out)
```

LayerNorm对每个样本、每个通道的特征维度计算 `γ(y−μ)/sqrt(σ²+ε)+β`。它沿特征求统计，与前面BatchNorm跨展开批次和时间求统计的方式不同。

训练时Dropout可写为 `Dropout(y)=r⊙y/(1−p)`，每个掩码值 `r` 以概率 `1−p` 为1，否则为0；`⊙`是对应位置相乘。保留值会缩放，评估时则直接保留输入。当前多头GAT包装层的Dropout默认为0.1。

### 12.8 通道注意力汇总：32份特征合成一份

GAT更新每个通道；接下来的通道注意力汇总，把32份通道特征变成一份代表整个一秒窗口的特征。

先用一个小型网络给每个通道算分数，再用 `softmax` 转换为总和为1的权重，最后按权重逐位置相加。例如：

```text
A的特征：[2, 4]，权重0.5
B的特征：[6, 8]，权重0.3
C的特征：[1, 3]，权重0.2

汇总结果：
[0.5×2+0.3×6+0.2×1, 0.5×4+0.3×8+0.2×3]
= [3, 5]
```

当前代码对32个通道这样汇总。一批从 `(32, 32, 64)` 变成 `(32, 64)`。每个样本现在有一份64维窗口特征，还没有输出类别。

**通道汇总的公式与代码**

$$
s_m=w_2^\top\tanh(W_1h_m+b_1)+b_2,\qquad
\beta_m=\frac{\exp(s_m)}{\sum_{j=1}^{M}\exp(s_j)},\qquad
z=\sum_{m=1}^{M}\beta_mh_m
$$

这里 `h_m` 是GAT后的64维通道特征；打分网络先从64维变到32维，用 `tanh` 激活，再变成一个分数。`β_m`是通道权重，`z`是整个窗口的64维表示。

```python
# 项目摘录：model_optimized.py，ChannelAttentionReadout
self.attention_scorer = nn.Sequential(
    nn.Linear(in_dim, in_dim // 2),
    nn.Tanh(),
    nn.Linear(in_dim // 2, 1),
)
scores = self.attention_scorer(node_features).squeeze(-1)  # (B,32)
weights = F.softmax(scores, dim=1).unsqueeze(-1)          # (B,32,1)
readout = (node_features * weights).sum(dim=1)            # (B,64)
```

`weights`最后一维为1，因此一个通道的同一个权重会乘到该通道全部64个特征上；`sum(dim=1)`消去通道维度。

### 12.9 调整特征维度，准备Transformer输入

用全连接层把每份64维窗口特征转换为128维。这里的128是模型设置的特征维度，不是原始脑电采样点。

Transformer按序列接收数据。如果一个样本有连续三秒，可以组织成三个位置；但当前入口使用默认 `n_slices=1`，每个样本只有一个位置：

```text
样本1 → [它自己的一个位置，128个特征]
样本2 → [它自己的一个位置，128个特征]
……

一批的形状：(32, 1, 128)
              ↑  ↑   ↑
            样本数 序列长度 特征数
```

代码还加上可学习的位置表示。当前只有一个位置，没有把同一次试验的60个窗口连成时间序列。批次中的32个样本也不会变成32个时间位置。

**投影和位置表示的公式与代码**

$$
U_s=W_pz_s+b_p,\qquad E_s=U_s+P_s
$$

`z_s`是第 `s` 个位置的64维窗口特征，`W_p`将它转换为128维；`P_s`是可学习的位置向量；`E_s`是Transformer输入。当前只有 `s=1`。

```python
# 项目摘录：model_optimized.py，GTENet.forward
Z = torch.stack(z_list, dim=1)               # (B,1,64)
Z_proj = self.readout_proj(Z)                # (B,1,128)
Z_pos = Z_proj + self.pos_emb.unsqueeze(0)   # (B,1,128)
```

`self.readout_proj`是 `nn.Linear(self.gat2_out_dim, d_model)`，冻结配置为 `Linear(64,128)`；`pos_emb`形状 `(1,128)`，增加批次维后可加到所有样本上。

还需注意：`_slice_windows`是在一个输入窗口内部切片，不是把不同样本或相邻秒自动连接。只把 `n_slices` 改大，并不等于建立连续多秒样本；当前也没有验证这种改法。

### 12.10 Transformer怎样处理特征

Transformer编码层主要包含自注意力和前馈网络，还有残差连接、归一化和Dropout。

自注意力让序列中的不同位置按权重交换信息。若有三秒，它可以计算更新第1秒时应参考第2秒、第3秒多少信息。但当前只有一个位置，没有其他秒可参考。在评估时，每个头的唯一注意力权重为1；训练时注意力Dropout可能对它做随机丢弃和缩放。这一环节仍有特征变换，不能因此说模块完全没有计算作用，但不能宣称它已经学习跨秒变化。

前馈网络进一步组合每个位置的特征。冻结配置中是：

```text
128维 → 全连接层 → 512维 → GELU → 全连接层 → 128维
```

中间的512维是模型加工特征的空间，不是新增的脑电测量。当前配置使用1层Transformer、4个注意力头；处理后形状仍为 `(32, 1, 128)`，特征值已发生变化。

**自注意力公式：Q、K、V分别做什么**

省略偏置和Dropout，单个头的核心计算是：

$$
Q=EW_Q,\quad K=EW_K,\quad V=EW_V,\qquad
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),\quad O=AV
$$

`E`是输入序列；`Q`可以理解为每个位置拿来询问的表示，`K`是用来匹配的表示，`V`是被汇总的信息。它们都是输入经不同可学习投影得到的特征，不是另三份数据。`QKᵀ`计算位置对分数，除以 `sqrt(d_k)`调整尺度，softmax沿提供信息的位置归一化。

当前4个头把128维分为每头32维，`d_k=32`。单头、单样本的 `Q/K/V` 为 `(1,32)`，注意力矩阵 `A` 为 `(1,1)`；没有第二个位置，评估时 `softmax([一个分数])=[1]`。多头结果拼接后再做输出投影。

前馈网络核心为：

$$
\operatorname{FFN}(h)=\operatorname{GELU}(hW_1+b_1)W_2+b_2
$$

矩阵写法采用行向量约定，`W_1`把128维映射到512维，`W_2`映射回128维。省略Dropout后，当前默认后归一化编码层可概括为：

$$
H_1=\operatorname{LayerNorm}(E+\operatorname{MHA}(E)),\qquad
H=\operatorname{LayerNorm}(H_1+\operatorname{FFN}(H_1))
$$

`MHA`是多头自注意力。实际PyTorch层还在注意力、前馈和残差路径使用Dropout。

```python
# 项目摘录：model_optimized.py，GTENet.__init__与forward
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=transformer_heads,
    dim_feedforward=ffn_dim, dropout=dropout,
    activation='gelu', batch_first=True,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
H = self.transformer(Z_pos)
```

`batch_first=True`约定输入顺序是 `(B,S,D)`。冻结配置是128维、4头、前馈512维、1层；不要与文件末尾随机张量示例中的8头、3层、多切片配置混淆。

### 12.11 序列汇总，再由分类器输出分数

模型设有时间注意力汇总，用来按权重合并序列位置。当前只有一个位置，权重为1，直接得到它的特征，一批从 `(32, 1, 128)` 变为 `(32, 128)`。

分类器继续处理：

```text
128维 → 全连接层 → 64维 → GELU → Dropout
      → 全连接层 → 2个类别分数
```

例如某个样本输出类别0分数为0.7、类别1分数为2.1，就预测为1，因为类别1分数更高。这些原始分数叫 `logits`，不是直接的概率。

32个样本各输出2个分数，一批输出形状为 `(32, 2)`。从输入到这些分数，叫前向传播；此时还没有更新参数。

**序列汇总和分类公式、代码**

时间汇总与通道汇总类似，只是加权的对象变成序列位置：

$$
s_s=q^\top(W_hH_s+b_h),\quad
\rho_s=\frac{\exp(s_s)}{\sum_{j=1}^{S}\exp(s_j)},\quad
v=\sum_{s=1}^{S}\rho_sH_s
$$

`q`是可学习查询向量；当前 `S=1`，所以 `ρ_1=1`、`v=H_1`。该单位置设置下，打分网络没有多个位置可供选择。

分类器写成：

$$
\ell=W_2\operatorname{Dropout}\bigl(\operatorname{GELU}(W_1v+b_1)\bigr)+b_2,
\qquad \widehat{y}=\arg\max_{c\in\{0,1\}}\ell_c
$$

这里采用列向量写法，`ℓ`是两个logits，`ŷ`是预测类别；`argmax`取最大分数所在的类别编号。

```python
# 项目摘录：model_optimized.py，TemporalAttentionPooling.forward
scores = torch.matmul(self.proj(H_seq), self.query)
alpha = F.softmax(scores, dim=1).unsqueeze(-1)
out = (H_seq * alpha).sum(dim=1)

# GTENet中的分类器定义与调用
self.classifier = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, num_classes),
)
H_pool = self.temp_pool(H)
logits = self.classifier(H_pool)
```

`num_classes=2`来自DEAP训练入口。模型类自身默认是3类，不能直接用所有默认参数代表本次任务。

### 12.12 用标签计算损失

损失衡量预测分数偏离训练目标的程度，不只是看对错。为了理解，先看没有标签平滑的普通交叉熵。

假设真实标签是1，把示意分数 `[0.7, 2.1]` 通过 `softmax` 转为概率，类别1约占80%。普通交叉熵的核心是：

```text
损失 = -log(正确类别的预测概率)
```

| 正确类别的预测概率 | 普通交叉熵损失，约值 |
| --- | ---: |
| 90% | 0.11 |
| 50% | 0.69 |
| 10% | 2.30 |

越支持正确类别，普通交叉熵越小。当前 `CrossEntropyLoss` 直接接收logits，在内部完成相关计算，不需要手动先做softmax。

项目使用 `label_smoothing=0.1`，二分类目标会从完全偏向一个类别，改成正确类别0.95、另一类别0.05，再计算损失。因此实际损失不是上表的普通交叉熵数值，也不是让正确类别概率无限接近1就始终更好。

每个样本分别计算损失，再对这一批取平均。到这里，只算出了一个损失值，还没有改变模型参数。

**softmax、标签平滑与批次损失的完整关系**

$$
p_{bc}=\frac{\exp(\ell_{bc})}{\sum_{j=0}^{C-1}\exp(\ell_{bj})},\qquad
q_{bc}=(1-\varepsilon)\mathbf{1}[c=y_b]+\frac{\varepsilon}{C}
$$

$$
L=-\frac{1}{B}\sum_{b=1}^{B}\sum_{c=0}^{C-1}q_{bc}\log p_{bc}
$$

`b`是样本编号，`c`是类别，`C=2`，`ε=0.1`；指示符 `1[c=y_b]`在类别正确时为1，否则为0。`p`是预测概率，`q`是平滑目标。例如标签1对应 `q=[0.05,0.95]`。`ε=0`时恢复普通交叉熵。

```python
# 项目摘录：GTE_train.py，train
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
logits = model(x_time, x_freq)       # (B,2)，浮点分数
loss = criterion(logits, labels)    # labels=(B,)，整数类别；loss是标量
```

输出概率只是理解公式的一种方式，项目把原始logits直接传给损失函数。不要先手动softmax再传入 `CrossEntropyLoss`。

### 12.13 反向传播算梯度，优化器更新参数

卷积、全连接层、GAT等模块都有可学习参数。反向传播从损失沿计算过程往回算，为每个参数计算梯度。

梯度可以理解成：这个参数稍微增大时，损失往哪个方向变化、变化多敏感？

用一个简化例子理解：`预测值=输入×权重`，输入为2、权重为0.3时预测为0.6。如果目标是1，适当增大权重能使预测更接近目标。这里只是解释调整方向，真实模型使用前述分类损失和更复杂的计算。

反向传播只算梯度，优化器才更新参数。最基础的梯度下降思路是：

```text
新参数 = 旧参数 − 学习率 × 梯度
```

学习率控制调整幅度。当前使用AdamW，会结合梯度历史和权重衰减更新参数，不是只按上面这条简化公式计算。更新前还做梯度裁剪，限制梯度总范数。

代码里的对应关系是：

```python
# 示意训练核心；实际脚本还有设备迁移、数据增强和指标统计
optimizer.zero_grad()               # 清空上一批梯度
logits = model(x_time, x_freq)       # 前向传播
loss = criterion(logits, labels)     # 计算损失
loss.backward()                     # 反向传播，计算梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()                    # 更新参数
```

这段解释训练逻辑，不是独立可运行的完整脚本。单次更新不保证下一批预测或验证表现一定改善，需要实际检查训练趋势和验证结果。

**梯度和链式法则：从损失往前追溯**

$$
g_r=\frac{\partial L}{\partial\theta_r},\qquad
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial u}\frac{\partial u}{\partial w}
$$

`θ_r`是第 `r` 个参数；偏导数描述只改变该参数时损失的局部变化。第二式是单条路径的链式法则，真实网络存在多条路径时，要把各路径贡献加起来。

教学例子：`u=wx`、`L=(u−a)²/2`，则 `∂L/∂w=(u−a)x`。取 `x=2,w=0.3,a=1`，梯度为 `(0.6−1)×2=−0.8`。若用普通梯度下降、学习率0.1，新权重为 `0.3−0.1×(−0.8)=0.38`。这用的是示意平方损失，不是当前项目的分类损失，也不是AdamW的完整更新。

AdamW的主要计算可写为：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
$$

这里 `t`是参数更新次数，不是epoch；`g_t`是裁剪后梯度，`m/v`积累梯度与平方梯度历史，帽号表示偏差校正，`η`是学习率、`λ`是权重衰减、`ε`是数值保护项。这些运算按参数位置分别执行；不必现在推导，只需知道AdamW不仅看本批梯度。

```python
# 项目摘录：GTE_train.py，从头训练分支
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
```

冻结初始 `lr=1e-4`、`weight_decay=1e-4`。未显式改动的AdamW参数采用运行环境中PyTorch的默认设置；学习率还会被调度器改变，续训分支需单独核对。

### 12.14 重复处理各批，完成一个epoch

处理完一批，拿下一批重复“前向计算、算损失、算梯度、更新参数”。训练集全部遍历一次，叫一个epoch。

窗口数量来自前面的切窗，不是模型额外产生的：

```text
一次试验：60秒 → 60个一秒窗口
一个人：40次试验 × 60 = 2,400个窗口
全部32人：32 × 2,400 = 76,800个窗口
```

前三秒基线不计入窗口。76,800个片段不等于76,800次独立试验；同次试验内窗口共享标签，相邻窗口可能很相似。

全量数据、80/20划分、batch size为32时：

```text
训练集：61,440个窗口 ÷ 每批32个 = 1,920批
验证集：15,360个窗口 ÷ 每批32个 = 480批
```

所以一个epoch训练阶段通常更新1,920次参数，不是只更新一次。这是配置对应的数量，不表示已经实际跑完了一个epoch。当前训练阶段还可能做随机脑电数据增强；这里主要解释核心流程。

**批次数公式与训练循环**

$$
N_{batches}=\left\lceil\frac{N_{train}}{B}\right\rceil
$$

`⌈ ⌉`表示向上取整，当前DataLoader没有丢弃末尾不足一批的样本。此处61,440恰好能被32整除。

```python
# 项目摘录：GTE_train.py，train；批内其他计算见12.13
for epoch in range(args.epochs):
    model.train()
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    for x_time, x_freq, labels in train_pbar:
        # 此处执行前向、损失、反向传播、梯度裁剪、更新
        ...
```

省略号是教学占位，不能用这段代替完整训练脚本。`model.train()`切换训练模式，并不自行运行训练，也不重新初始化模型；真正的参数更新由批内的 `optimizer.step()` 完成。

### 12.15 用验证集检查，记录和保存结果

一个epoch训练完，使用当前参数预测验证集。代码调用 `model.eval()`，使Dropout停止随机丢弃，BatchNorm使用已积累的运行统计；在 `torch.no_grad()` 中计算预测，不进行反向传播，也不更新参数。

验证Accuracy按全部验证窗口统计：

```text
验证Accuracy = 验证集中预测正确的窗口总数 ÷ 验证窗口总数
```

记录训练和验证损失、准确率。当前脚本根据验证损失调整学习率；验证准确率提高时保存检查点。检查点包含模型参数、优化器状态等，不是只保存一个准确率数字。随后进入下一epoch，直到达到设置轮数或触发提前停止，训练结束还会输出曲线。

**准确率公式与验证代码**

$$
\operatorname{Accuracy}_{val}
=\frac{1}{N_{val}}\sum_{i=1}^{N_{val}}\mathbf{1}[\widehat y_i=y_i]
$$

指示符预测对时为1，否则为0；统计范围是全部验证窗口，不是取某一批的准确率，也不是逐受试者平均准确率。

```python
# 项目摘录：GTE_train.py，验证核心
model.eval()
correct_preds = 0
total_preds = 0
with torch.no_grad():
    for x_time, x_freq, labels in val_pbar:
        x_time, x_freq, labels = x_time.to(device), x_freq.to(device), labels.to(device)
        logits = model(x_time, x_freq)
        loss = criterion(logits, labels)
        _, predicted = torch.max(logits.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
val_acc = correct_preds / total_preds
```

`model.eval()`控制Dropout和BatchNorm等层的行为；`no_grad()`关闭这段计算的梯度记录，二者不是同一件事。`torch.max(...,1)`沿类别维取最大值，第二个返回值是类别索引。

```python
# 项目摘录：GTE_train.py，选择与保存相关核心语句
scheduler.step(val_loss)
if val_acc > best_val_acc:
    best_val_acc = val_acc
    # 实际脚本还记录最佳训练准确率、生成文件名、保存优化器等状态
    ...
```

学习率调度监控 `val_loss`，最佳检查点监控 `val_acc`，是两个不同依据。这里只解释当前行为，不表示保存与完整训练链路已经在本次操作中运行验证。

### 12.16 反复验证会不会发生数据泄漏

需要区分两种情况。

**第一种是每个epoch使用同一个验证集。** 这是常规模型选择流程，验证时不反向传播。但验证结果会影响学习率、停止时机、检查点选择和后续调参，所以验证集参与了模型选择。反复根据验证结果修改方案，也可能逐渐迎合这份验证集。最终需要一份未参与这些选择的独立测试集；当前窗口级流程没有这份测试集。

**第二种是当前按窗口随机划分。** 同一个人同一次试验的第8秒可能进入训练集，第9秒进入验证集。信号和背景可能相似，模型已经见过这个人和这次试验的相关信息，验证结果可能偏乐观。这是当前流程的重要信息泄漏风险；不做验证反向传播也不能消除它。

若评估新受试者，应按人划分，例如：

```text
训练集：一组人的全部窗口 → 学习参数
验证集：另一组人的全部窗口 → 选择模型和方案
测试集：再一组人的全部窗口 → 方案确定后的最终评估
```

三组人互不重叠，归一化等需要从数据估计的处理参数也只能使用训练集确定。这是后续评估原则，不是当前已实现的代码。当前先用窗口级划分核对上游流程，其结果不能代表对新人的泛化能力。

**当前划分代码与未来检查条件**

```python
# 项目摘录：GTE_train.py，当前窗口级划分
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=generator
)
```

随机种子固定的是划分，在样本顺序和数量相同时可重复；不代表模型初始化、数据增强等全部随机性已固定。

按受试者划分的必要条件可以写为：

$$
S_{train}\cap S_{val}=\varnothing,\quad
S_{train}\cap S_{test}=\varnothing,\quad
S_{val}\cap S_{test}=\varnothing
$$

这里 `S_train/S_val/S_test`是三组受试者ID集合（不是前面的序列长度 `S`）；`∩`表示交集，`∅`表示空集。意思就是任意两组没有同一个人。未来实现时需要保留样本身份并检查这些条件，不能仅把随机种子或划分比例改掉就声称完成。

### 12.17 把整条流程连起来

```text
一批配对的脑电、DE和标签
    │
    ├─ 脑电 → 多尺度一维卷积 → 沿时间平均 ─┐
    └─ DE   → 全连接层 + GELU ────────────┤
                                          ↓
                              每个EEG通道的特征拼接
                                          ↓
                                  两层多头GAT
                                          ↓
                                  通道注意力汇总
                                          ↓
                              全连接投影、单位置序列
                                          ↓
                                     Transformer
                                          ↓
                                  时间注意力汇总
                                          ↓
                                    分类器、logits
                                          ↓ 与标签比较
                                        损失
                                          ↓
                              反向传播、梯度裁剪、AdamW
                                          ↓
                                    下一批训练样本

训练集遍历一次 → 验证 → 记录、选检查点 → 下一epoch
```

学习时可以停在每个小节，先解释输入是什么、这一步做什么、输出是什么，再继续。完整理解不要求一次背下所有术语。

**整条前向传播的形状速查**

| 环节 | 一批数据的形状 | 核心代码位置 |
| --- | --- | --- |
| 脑电输入 | `(B,32,128)` | `GTE_train.py`：`DEAPDataset`、`DataLoader` |
| DE输入 | `(B,32,5)` | 同上，与脑电和标签配对 |
| 卷积特征 | `(B,32,64,128)` | `model_optimized.py`：`Conv1DBankShared` |
| 波形时间平均 | `(B,32,64)` | `_spatial_pipeline`：`mean(dim=-1)` |
| DE投影 | `(B,32,64)` | `freq_proj` |
| 拼接 | `(B,32,128)` | `_spatial_pipeline`：`torch.cat` |
| 两层GAT后 | `(B,32,64)` | `gat1/ln1`、`gat2/ln2` |
| 通道汇总 | `(B,64)` | `ChannelAttentionReadout` |
| 组织单位置序列 | `(B,1,64)` | `forward`：`torch.stack(z_list,dim=1)` |
| 投影与位置表示 | `(B,1,128)` | `readout_proj`、`pos_emb` |
| Transformer | `(B,1,128)` | `transformer` |
| 时间汇总 | `(B,128)` | `TemporalAttentionPooling` |
| 类别分数 | `(B,2)` | `classifier` |
| 标签、批次损失 | `(B,)`、标量 | `GTE_train.py`：`criterion` |

本表对应冻结配置。改头数、输出维度、窗口长度或切片设置后，必须重新核对，不能直接照搬这些数字。

完成标准：能区分批次与时间序列、采样点与特征值、前向传播与参数更新、一个批次与一个epoch，以及窗口级验证与新受试者评估。实际运行仍按 `docs/REPRODUCTION_PROTOCOL.md` 第7.3节先做1 epoch冒烟实验；本文说明不是已完成的实验记录，也不新增准确率结果。

## 第十三步 运行训练前检查

第十二步讲清了模型怎样计算和训练。现在动手运行一次检查，确认当前电脑上的环境、数据读取、真实窗口的特征计算和模型前向传播能正常执行。

本节记录的是操作步骤，不表示已经执行或通过检查。依据是 `docs/REPRODUCTION_PROTOCOL.md` 第7.3节及 `validate_setup.py`。

### 13.1 打开PowerShell，进入项目文件夹

在Windows开始菜单搜索并打开PowerShell，输入下面命令，按回车：

```powershell
Set-Location 'D:\workspace\GTE-Net'
```

`Set-Location`用于切换当前目录。后面的检查命令要在项目根目录下执行。

### 13.2 运行检查脚本

输入下面命令，按回车：

```powershell
.\.venv\Scripts\python.exe validate_setup.py
```

- `.\.venv\Scripts\python.exe`：使用本项目虚拟环境中的Python。
- `validate_setup.py`：执行项目已有的检查脚本。

不需要把第十二步的代码摘录全部复制进PowerShell，也不需要为这次检查重新编写模型。

### 13.3 对照输出，判断是否通过

正常输出应包含：

```text
OK: 32 DEAP files
OK: data=(40, 40, 8064), labels=(40, 4)
OK: 60 one-second stimulus windows, baseline excluded
OK: real EEG features time=(32, 128), freq=(32, 5)
OK: model forward output=(1, 2)
```

| 输出 | 表示什么 |
| --- | --- |
| `32 DEAP files` | 预期的32个受试者文件已找到 |
| `data`和`labels`形状 | 抽样读取的 `s01.dat` 形状符合预期，不代表已读取检查全部32个文件的内部内容 |
| `60 one-second stimulus windows` | 一次试验的60个一秒窗口位置检查通过，基线未计入刺激期窗口 |
| `time`和`freq`形状 | 一个真实窗口的基线校正和五频带DE特征计算通过相应检查 |
| `model forward output=(1, 2)` | 一个样本通过模型后得到两个类别分数，不表示预测已经准确 |

脚本还会显示当前可用设备为 `cpu` 或 `cuda`。显示 `cpu` 不代表检查失败；它说明当前运行环境没有检测到可用CUDA。

等待脚本结束、PowerShell重新出现输入提示符后，可以输入：

```powershell
$LASTEXITCODE
```

如果上述检查输出齐全、没有报错，且退出码为 `0`，本次检查通过。这里只说明检查范围内的流程可用，不代表完整训练已完成。

### 13.4 如果报错，先停在这一步

保留执行的命令和完整报错文本，尤其是最后的错误类型与信息，不要只截取一行。先解决问题并重新运行检查，通过后再进入下一步。

例如，找不到Python路径时先核对当前目录和 `.venv` 是否存在；出现缺少模块或数据文件的错误时，根据完整报错检查对应依赖或路径，不要删除原始数据或随意改动预处理定义。

### 13.5 明确这一步完成了什么

这个脚本只抽样读取真实数据、计算一个窗口的特征并执行模型前向检查，不生成全量缓存，也不进行完整训练。

完成标准：脚本正常结束，预期检查输出齐全，退出码为 `0`。实际运行后，再把命令、环境和真实输出追加到 `docs/EXPERIMENT_LOG.md`；不要把本节说明当作已完成的实验记录。

下一步才是第十四步：使用冻结命令运行1 epoch，实际检查全量预处理、训练、验证和保存流程。


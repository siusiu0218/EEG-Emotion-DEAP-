# GTE-Net 数据目录说明

整理日期：2026-08-28

## 当前用于复现的数据

`dataset/DEAP/data_preprocessed_python/`

- 包含 `s01.dat` 至 `s32.dat`，共 32 个文件。
- 这是 GTE-Net 复现时优先使用的 DEAP Python 预处理版本。
- 抽样读取 `s01.dat` 已通过：`data` 形状为 `(40, 40, 8064)`，`labels` 形状为 `(40, 4)`。
- 解压后的数据约 3.076 GB。

## 原始下载包

所有压缩包均保留在 `datas/archives/`，没有删除。

### DEAP

- `DEAP/required/data_preprocessed_python.zip`：当前复现需要，已解压到正式数据目录。
- `DEAP/optional/data_preprocessed_matlab.zip`：Matlab 预处理版，当前代码路线暂不使用。
- `DEAP/optional/data_original.zip`：DEAP 原始信号，当前复现暂不使用。
- `DEAP/optional/face_video.zip`：面部视频，GTE-Net EEG 训练不使用。
- `DEAP/documentation/`：元数据和数据说明文档。

### SEED

- `SEED/primary/SEED.zip`：标准 SEED 数据，留作后续实验。
- `SEED/other_variants/SEED_IV.zip`：SEED-IV，不等同于标准 SEED。
- `SEED/other_variants/SEED-VIG.zip`：SEED-VIG，不是当前情绪分类复现数据。

## 使用原则

1. 第一阶段只使用 `dataset/DEAP/data_preprocessed_python/`。
2. 暂不解压 DEAP 原始数据、视频、Matlab 版本及 SEED 系列，以免混用并节省空间。
3. 后续补入代码时，让 DEAP 数据读取路径指向上述正式数据目录。

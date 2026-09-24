# 问题1 多模态特征提取与时序对齐

> **版本提示（2026-09-23）**：本文件下述既有说明针对旧版 `output/`，其文本采用整段均匀弱对齐。新增的声学约束版本存于 `output_speech_aware/`，保持旧版不变；不要混用两版的清单与数字。新版方案、静音策略及复现命令详见 `speech_alignment_report.md`，入口为 `run_speech_alignment.ps1`。原始语音有效性 `audio_mask`、声学活动候选 `speech_mask` 与实际可定位文本 `text_mask` 是三种不同含义。

本目录完成附件1全部100条视频的文本、语音、视觉三模态特征提取与统一时序对齐。最终文件采用与附件2对齐版一致的核心尺寸：文本 `50×768`、语音 `50×74`、视觉 `50×35`。此外保存时间边界、三模态有效掩码、源文件哈希、处理日志和典型样本逐窗核验表。

## 1 数学模型

对时长为 `T` 的样本建立 `K=50` 个公共时间窗：

```text
I_k = [kT/K, (k+1)T/K),  k=0,...,K-1
```

最后一个时间窗包含右端点 `T`。模态原始特征在公共时间窗上的表示定义为：

```text
z_k^(m) = sum_i |I_k ∩ J_i^(m)| x_i^(m) / sum_i |I_k ∩ J_i^(m)|
```

其中 `J_i^(m)` 是模态 `m` 的第 `i` 个局部观测所覆盖的原始时间区间，`x_i^(m)` 为该观测的特征。若窗口内没有有效观测，则特征置零，同时对应的 `modality_mask[k]=0`。后续模型必须联合读取特征与掩码，不能把填充零当成真实信号。

### 文本 768维

- 工具：`distilbert-base-uncased`，`transformers 4.29.2`。
- 表示：最后4层隐状态取均值，得到每个词元的768维上下文特征。
- 时间定位：题目转写没有逐词时间戳，因此采用单调文本分配。单词时长权重为字符数的 `0.70` 次幂，逗号类停顿权重为 `0.35`，句末停顿权重为 `0.70`；累计权重线性映射到 `[0,T]`。BERT子词继承其所属单词的时间段，再按与 `I_k` 的重叠时长加权池化。
- 该规则不声称是强制对齐结果；它是无逐词时间戳条件下可复现、单调且可核验的弱对齐假设。`text_alignment.jsonl` 保留全部词元区间，便于论文说明和人工复核。

### 语音 74维

- 音频统一重采样为16 kHz、单声道。
- 每个公共窗口内使用25 ms帧长、10 ms帧移计算：20维MFCC的均值和标准差（40维）、12维Chroma均值、8类频谱描述量的均值和标准差（16维）、基频均值/标准差（2维）、过零率均值/标准差（2维）、RMS能量均值/标准差（2维），共74维。
- 8类频谱描述量为频谱质心、带宽、85%滚降点、平坦度、谱熵、谱通量、低频能量比和高频能量比。

### 视觉 35维

- 每个公共窗口选择最接近窗口中心的原始视频帧。
- OpenCV Haar人脸检测生成6维人脸存在性和归一化几何特征。
- 全帧强度统计6维、HSV颜色统计6维、边缘/梯度/纹理5维、相邻对齐帧稠密光流4维、人脸区域或中心区域的分区亮度与左右对称性8维，共35维。
- 未检测到人脸不等于视觉缺失：人脸存在特征为0，但全帧外观和运动特征仍有效。只有没有可解码帧时 `vision_mask=0`。

## 2 输出文件

运行后在 `output` 目录生成：

- `question1_aligned_50.pkl`：100条样本的三模态特征、标签、时间边界、掩码和元数据。
- `feature_manifest.csv`：300行，即100条样本乘3个模态；含源文件、时长、维度、粒度、有效窗口、填充规则、提取器和SHA-256。
- `processing_log.jsonl`：逐样本处理耗时、解码帧数和有效窗口数。
- `text_alignment.jsonl`：文本词元的字符区间、秒级区间及其对应时间窗。
- `typical_sample_alignment.csv`：自动选取的典型样本逐窗核验表，可直接用于论文中的对齐案例表或绘图。
- `typical_sample_frames.csv` 与 `typical_sample_frames/`：典型样本5个代表时间窗的原始视频帧及秒级索引。
- `quality_report.json`：形状、有限值、覆盖率、标签分布和最终文件哈希。

PKL读取示例：

```python
import pickle

with open("output/question1_aligned_50.pkl", "rb") as handle:
    payload = pickle.load(handle)

data = payload["all"]
print(data["text"].shape)        # (100, 50, 768)
print(data["audio"].shape)       # (100, 50, 74)
print(data["vision"].shape)      # (100, 50, 35)
print(data["time_bounds"].shape) # (100, 50, 2)
```

## 3 环境与运行

当前机器采用两个隔离环境：

- 音视频环境：Python 3.12，安装 `requirements-av.txt`。
- 文本环境：现有 `D:\Anaconda3\envs\pytorch`，PyTorch 1.8.0 + CUDA 11.1；把 `requirements-text.txt` 安装到临时目录或该环境。

当前可直接执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_alignment.ps1
```

若重新创建临时依赖：

```powershell
$avEnv = Join-Path $env:TEMP "codex_q1_alignment_env"
& "C:\Users\起飞\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv --system-site-packages $avEnv
& "$avEnv\Scripts\python.exe" -m pip install -r .\requirements-av.txt

$textDeps = Join-Path $env:TEMP "codex_q1_pytorch_packages"
& "D:\Anaconda3\envs\pytorch\python.exe" -m pip install --target $textDeps -r .\requirements-text.txt
```

`run_alignment.ps1` 支持 `-DataRoot`、`-OutputDir`、`-AvPython`、`-BertPython` 和 `-TransformersPath` 参数，用于迁移到其他机器。

## 4 核验规则

`verify_outputs.py` 自动检查：样本数与ID唯一性、三模态精确形状、NaN/Inf、掩码与零填充一致性、50个时间窗连续性、末端与视频时长一致性，以及汇总表是否完整包含300条“样本-模态”记录。

标签仅随原始特征保存，不参与问题1特征提取或参数选择。整个流程没有引入额外情感数据集。

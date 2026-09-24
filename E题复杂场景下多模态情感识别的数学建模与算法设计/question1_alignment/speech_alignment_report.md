# 问题一声学约束对齐与静音样本处理报告

## 赛题要求与结论

赛题“数据说明”规定附件1所有内容由题目统一提供，**不得替换、增补、删除或修改样本及其标签**；对无效或异常片段可以掩码、截断或采取其他处理，但**必须保留原始样本并说明规则**。“问题1”要求100条原始样本覆盖完整、样本与模态文件及特征逐条对应，有效长度、填充和原素材时间关系可核验，且提供工具参数、日志、典型展示和复现步骤。题目**没有要求把静音视频删除，也没有允许在静音上编造词的真实发声时间**。开源工具和预训练模型允许用于基础特征提取；不得使用外部情感数据集参与本题训练或参数选择。本版使用已有英语ASR预训练模型作词级声学约束，未用外部情感数据集训练。

**对“有转写但音轨全零”的决策**：保留该视频在100条中的位置及其原始转写、分类和连续标签；保留视频帧特征和50个时间窗。音轨确实存在且可解码，故 `audio_mask` 表示“音频流可用”，可以是1；`speech_mask` 表示“检测到有声活动候选”，此时全为0；`text_mask` 表示“文本能被安全定位到此时间窗”，全为0，文本特征50窗填零。`alignment_status=transcript_audio_conflict` 和独立日志明确表明原始转写与音轨互相矛盾。**文本没有被删除，只是不把它强行赋予虚假的词级时间戳。** 不从情感标签推断语音状态，也不改变标签。

本数据中两个对应样本是 `-mJ2ud6oKI8$_$1` 和 `-mJ2ud6oKI8$_$2`。两者原音轨解码后均为全零，原Excel转写均非空，原始情感极性为Neutral、强度为0；两者输出 `audio_mask=50/50`、`speech_mask=0/50`、`text_mask=0/50`、`vision_mask=50/50`。按本题“保留并说明掩码”处理，比将转写照旧均摊到50窗更诚实。

## 新版实际对齐规则

1. 保留旧版 `work/` 的100条样本ID、媒体时长、原始标签、50窗和视觉特征。独立重解码原音轨；语音每窗只用**实际音频样本**计算74维特征，`audio_coverage[k]` 是该时间窗真实音频样本比例，消除旧版在音轨结尾混入人工补零的问题。音轨实际可用部分不足25毫秒时该窗填零并置 `audio_mask=0`。音轨可用但没有人说话，依然是可用的音频，不应简单设 `audio_mask=0`。
2. 在真实音频上每20毫秒求一次RMS，活动阈值 `max(0.003, 0.12×RMS第90百分位)`；合并极短间隔、保留至少约80毫秒片段，记录 `candidate_activity_segments_s`、每窗 `speech_coverage`/`speech_mask`。这是**有声活动候选，不是经过声学验证的准确人声VAD**；背景音乐/噪音也可能触发，弱语音也可能漏检。
3. 对非全零且有候选区间的音频，使用 `facebook/wav2vec2-base-960h` 的CTC帧发射概率，利用**赛题提供的原始转写**做Viterbi词级强制对齐。CTC路径中的英文词首尾帧转成秒，并与有声候选区间取交集，然后将768维DistilBERT词元特征按重叠秒数映射到50窗。无证据的窗口 `text_mask=0` 且文本特征全零。`text_alignment.jsonl` 保留每个词的字符范围、秒级时间与路径得分，以及每窗词列表。
4. 为避免不匹配文本被强制贴到背景音，CTC路径上各字符帧的**平均后验分数低于0.50**时，本版**保守拒绝整条样本的词级定位**，记 `ctc_low_confidence`，文本窗全部置空但保留 `raw_text`、标签与其余模态。此阈值是工程保守门槛，**不是词时间准确率或经过标注集校准的可靠概率**；人工听辨与后续抽查仍是必要的。

本次结果：**80条 `ctc_aligned_unverified`、18条 `ctc_low_confidence`、2条 `transcript_audio_conflict`**。样本仍是100条且原始标签未改变。文本有效窗2934/5000（58.68%）、语音有效窗4950/5000（99.0%）、视觉4990/5000（99.8%）；有声候选窗4337/5000（86.74%）。这些百分比是**时间窗的掩码比例**，不是经人工校准的词级定位准确率。80条仍标 `unverified`，不能声称所有词都准确；18条低置信样本的语音仍保留，可以复核或换更合适的预训练模型，不能当作无转写样本。

## 文件和复现

- 新版为 `output_speech_aware/question1_aligned_50.pkl`，顶层 `all`/`metadata`，三模态尺寸仍分别为 `(100,50,768)`、`(100,50,74)`、`(100,50,35)`。新字段为 `audio_coverage`、`speech_coverage`、`speech_mask`、`alignment_status`。`text_mask` 代表**文本定位有效**，不代表原始Excel是否有转写；可用 `raw_text` 查原文。
- `output_speech_aware/speech_activity.jsonl` 保存100条活动候选、初筛状态和最终对齐状态；`text_alignment.jsonl` 保存词时间及窗内词；`feature_manifest.csv` 含100×3条可核验记录；`quality_report.json` 汇总掩码和状态。旧版 `output/` 未覆盖，防止旧弱对齐与新版文件混用。
- 复现顺序：安装 `requirements-av.txt`、`requirements-text.txt`，准备PyTorch 1.8.0、Transformers 4.29.2、PyAV 18.1.0、OpenCV 4.10.0；运行 `./run_speech_alignment.ps1`，必要时显式传 `-DataRoot`、`-AvPython`、`-BertPython`、`-TransformersPath` 和 `-OutputDir`。脚本依次运行原视频特征准备（若 `work/base_features.pkl` 已存在则重用）、`prepare_speech_alignment.py`、`force_align_transcripts.py`、`extract_text_features.py`、`assemble_outputs.py`、典型帧导出及 `verify_outputs.py`。CTC模型修订号固定为 `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`，文本模型修订号为 `12040accade4e8a0f71eabdb258fecc2e7e948be`；首次运行需从模型仓库获取预训练权重，权重本身不属于≤50MB的结果附件。
- 试听示例位于 `review_examples_speech_aware/`：正常带词的样本、低置信拒绝样本、全零音轨冲突样本各1个，均保留原声；示例仅供人工复核，不能替代定量定位评价。视频里 `Predicted words` 是自动结果，不是人工真值。

**仍需改进**：本方法的声学活动门槛不是鲁棒语音识别，背景声、剪辑错配及部分语音过弱会影响路径；尤其18条低置信片段应人工试听或采用更强ASR对齐后再决定是否使用。视觉空窗与旧版所述相同，仍需按帧时间戳人工或程序检查；赛题并未要求引入额外情感语料解决这些问题。

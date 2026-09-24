# E题论文

## 主模型

论文最终主模型为统一接口的 attention--gate 层次融合模型，不使用旧 `residual fusion` 作为最终主模型。主模型固定使用：

- `text_bert` 输入、固定 `distilbert-base-uncased` revision；
- 音频、视觉和文本的 `aligned_50` 公共时间轴；
- 三路两层双向 GRU、掩码时间注意力、样本级模态门控；
- 分类与连续强度双任务输出，以及窗口遮挡解释。

## 编译

图表使用 `nature-figure` 技能的 Python/matplotlib 工作流和统一中文字体。固定基线用深灰、最终模型用深青；文本、语音和视觉分别使用橙、青、蓝，其他候选方案用紫色与金色区分。缺失场景的热力图使用以零为中心的对称对数色阶，数值标签仍为未经变换的真实分数差值。先运行
`D:\Anaconda3\envs\pytorch\python.exe make_statistical_figures.py`，再运行
`D:\Anaconda3\python.exe make_figures.py`。PPT 框图源文件由
`node make_ppt_diagrams.js` 生成，论文用矢量版本由
`node export_diagram_svgs.js` 生成，并以 `rsvg-convert` 导出同名 PDF；最后把两张 `*_ppt.pdf` 复制为 `fig_pipeline.pdf` 和 `fig_model_architecture.pdf`，以保持正文使用的框图与可编辑 PPT 一致。
数值图的逐样本结果与消融记录保存在 `figures/source_*.csv`、
`figures/source_representative_sample.json`；缺失热力图是统一编码候选的敏感性审计，
而非最终模型的缺失性能。视频证据仅在展示图中统一裁切画面下缘约 13% 以避开原片外文水印，原始关键帧和证据清单不做修改。所有验证集比较均为单次训练的描述性结果，不标记显著性。

附件4类别分布由最终模型的 `q3_predictions.csv` 直接统计，为消极 8 条、中性 3 条、积极 9 条，预测强度均值 -0.131304；不可混用附件3的均值。

在本目录执行三轮 XeLaTeX，并在第一次之后运行 BibTeX：

```powershell
A:\texlive\2026\bin\windows\xelatex.exe -interaction=nonstopmode main.tex
A:\texlive\2026\bin\windows\bibtex.exe main
A:\texlive\2026\bin\windows\xelatex.exe -interaction=nonstopmode main.tex
A:\texlive\2026\bin\windows\xelatex.exe -interaction=nonstopmode main.tex
```

最终 PDF 为 `main.pdf`，提交副本位于：
`A:\naiwa-modeling-team\output\pdf\多模态情感识别数学建模论文.pdf`。

## 结果口径

最终模型在附件 2 验证集相对固定对齐 baseline 的四项指标均改善：Accuracy `0.612637 -> 0.637363`、Macro-F1 `0.588115 -> 0.601580`、MAE `0.616443 -> 0.612262`、Pearson `0.624582 -> 0.629627`。固定 baseline 使用原始 `aligned_features.pkl`，最终模型使用 `aligned_text_bert_features.pkl`；验证划分相同，但文本特征不同，这一优势只能解释为完整系统比较，不能归因于架构单项。附件 2 test、附件 3 和附件 4 的限制及无标签边界在正文中单独报告，不将旧残差融合结果作为最终结论。

# E题论文

## 主模型

论文最终主模型为统一接口的 attention--gate 层次融合模型，不使用旧 `residual fusion` 作为最终主模型。主模型固定使用：

- `text_bert` 输入、固定 `distilbert-base-uncased` revision；
- 音频、视觉和文本的 `aligned_50` 公共时间轴；
- 三路两层双向 GRU、掩码时间注意力、样本级模态门控；
- 分类与连续强度双任务输出，以及窗口遮挡解释。

## 编译

图表使用 `nature-figure` 技能的 Python/matplotlib 工作流和统一中文字体。固定基线用深灰、最终模型用深青；文本、语音和视觉分别使用橙、青、蓝，其他候选方案用紫色与金色区分。缺失场景图直接标注真实 Macro-F1 变化（百分点），红色表示下降、绿色表示上升，颜色深浅仅辅助阅读。先运行
`D:\Anaconda3\envs\pytorch\python.exe make_statistical_figures.py`，再运行
`D:\Anaconda3\python.exe make_figures.py`。PPT 框图源文件由
`node make_ppt_diagrams.js` 生成，论文用矢量版本由
`node export_diagram_svgs.js` 生成，并以 `rsvg-convert` 导出同名 PDF；模型结构图使用 `fig_model_architecture.pdf`。总流程图以及三个问题分析小节的流程图使用 `make_three_question_roadmap.js` 绘制为四份可编辑 PPT，再由 `export_three_question_roadmap.ps1` 通过 PowerPoint 导出为论文中的 `fig_pipeline.pdf`、`fig_problem1_analysis.pdf`、`fig_problem2_analysis.pdf` 和 `fig_problem3_analysis.pdf`。这一步应在旧图表脚本之后运行，以免总流程图被旧版覆盖。
之后运行 `python make_revision_figures.py` 和 `python make_explanatory_figures.py`，重绘的统计图均直接取自保存的结果 CSV，不修改数据或重新推理。图内英文和数字使用 Times New Roman，中文保留中文字体。若缺少重新推理所需的派生特征文件，可运行 `python restyle_saved_svgs.py`，再运行 `powershell -ExecutionPolicy Bypass -File export_three_question_roadmap.ps1`，从已保存的矢量图导出相同数据、更新字体的 PDF。
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
`B:\houzez\naiwa-modeling-team\output\pdf\多模态情感识别数学建模论文.pdf`。

## 结果口径

最终模型在附件 2 验证集相对固定对齐 baseline 的四项指标均改善：Accuracy `0.612637 -> 0.637363`、Macro-F1 `0.588115 -> 0.601580`、MAE `0.616443 -> 0.612262`、Pearson `0.624582 -> 0.629627`。固定 baseline 使用原始 `aligned_features.pkl`，最终模型使用 `aligned_text_bert_features.pkl`；验证划分相同，但文本特征不同，这一优势只能解释为完整系统比较，不能归因于架构单项。附件 2 test、附件 3 和附件 4 的限制及无标签边界在正文中单独报告，不将旧残差融合结果作为最终结论。

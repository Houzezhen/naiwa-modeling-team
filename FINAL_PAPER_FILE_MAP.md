# 最终论文文件关系说明

本文件说明最终论文的源文件、数据、模型、图表和交付文件之间的关系。表内路径均相对于当前仓库根目录，不依赖电脑上的绝对路径。

## 1. 最终交付物

| 文件 | 作用 |
|---|---|
| `output/pdf/多模态情感识别数学建模论文.pdf` | 对外提交的最新 PDF；由 `2026华为杯研赛latex模板/E题论文/main.tex` 编译后复制得到 |
| `2026华为杯研赛latex模板/E题论文/main.tex` | 论文唯一主源文件，包含正文、公式、表格、图注、附录和复现实验命令 |
| `2026华为杯研赛latex模板/E题论文/reference.bib` | 参考文献数据库 |
| `2026华为杯研赛latex模板/E题论文/gmcmthesis.cls` | 论文模板类文件 |

`2026华为杯研赛latex模板/E题论文/main.pdf` 是本机编译产物；完成编译后将其复制到上述提交 PDF 路径。

## 2. 文件依赖关系

```text
E题复杂场景下多模态情感识别的数学建模与算法设计/
  └─ question1_alignment/                         问题1：时间对齐和特征审计
       └─ output_speech_aware/typical_sample_frames/  典型帧来源
                         │
emotion-transfer-pilot/  ─┼─ aligned_50.pkl          统一50窗多模态特征
                          └─ aligned_text_bert_features.pkl
                                      │
q2_baseline/                            │ 问题2：统一接口基线与候选模型
  └─ aligned_text_bert_run_bidir/       ├─ fusion_validation.json
       └─ fusion_best.pt                └─ 附件3专项推理结果
                                      │
q3_explainability/                      │ 问题3：最终复杂主模型与解释
  └─ q3_aligned_text_bert_final/        ├─ q3_explainable_best.pt
       ├─ validation.json               ├─ explanation_validation.json
       └─ attachment4/                  ├─ q3_predictions.csv
                                       ├─ q3_explanations.csv
                                       └─ video_keyframe_manifest.csv
                                      │
2026华为杯研赛latex模板/E题论文/figures/
  ├─ source_model_metrics.csv            数据来源与指标口径
  ├─ *.pdf                               论文实际插入的矢量图
  ├─ *.png / *.svg                       本地预览和矢量中间文件
  └─ 生成脚本                            │
                                      ▼
2026华为杯研赛latex模板/E题论文/main.tex
                                      │ XeLaTeX + BibTeX
                                      ▼
output/pdf/多模态情感识别数学建模论文.pdf
```

## 3. 图表生成链

| 文件 | 关系 |
|---|---|
| `2026华为杯研赛latex模板/E题论文/nature_plot_style.py` | 统一 Nature 风格、中文字体和高对比配色 |
| `2026华为杯研赛latex模板/E题论文/make_figures.py` | 生成流程图、模型比较、缺失模态、解释卡、附件4和复杂度图 |
| `2026华为杯研赛latex模板/E题论文/make_statistical_figures.py` | 生成主模型性能、热力图、混淆矩阵、回归散点图、时间注意力和消融图 |
| `2026华为杯研赛latex模板/E题论文/make_revision_figures.py` | 最后重绘候选模型、基线对照和验证/测试变化三张比较图，并校验图表数值与论文表格 |
| `2026华为杯研赛latex模板/E题论文/make_explanatory_figures.py` | 从原始CSV重绘混淆矩阵、缺失位置热力图和附件4分布；仅改变表达，不运行推理或修改数据 |
| `2026华为杯研赛latex模板/E题论文/make_ppt_diagrams.js` | 绘制模型结构图的可编辑 PPT 对象 |
| `2026华为杯研赛latex模板/E题论文/export_diagram_svgs.js` | 将原模型结构图 PPT 导出为论文使用的 PDF/PNG |
| `2026华为杯研赛latex模板/E题论文/make_three_question_roadmap.js` | 参照用户提供的流程图模板，为总路线及三问的问题分析生成四份可编辑 PPT |
| `2026华为杯研赛latex模板/E题论文/export_three_question_roadmap.ps1` | 使用 PowerPoint 从上述四份 PPT 精确导出 PDF/PNG，其中总路线覆盖 `fig_pipeline.pdf` |
| `2026华为杯研赛latex模板/E题论文/figures/` | `main.tex` 通过 `\\includegraphics` 插入的最终图形目录 |

其中，`make_revision_figures.py` 与 `make_explanatory_figures.py` 在其他统计图脚本之后依次执行；源数值不变。模型结构图仍由原 PPT 生成链提供。总流程图及三张问题分析图由新增的 PPT 生成链在最后导出，因此 `fig_pipeline.pdf` 与正文引用的三张分析图均可回溯到对应的可编辑 PPT。

## 4. 报告和证据文件

| 文件 | 作用 |
|---|---|
| `E题复杂场景下多模态情感识别的数学建模与算法设计/三个问题综合审查与最终结果报告.md` | 三个问题的总体路线、风险和最终指标口径 |
| `E题复杂场景下多模态情感识别的数学建模与算法设计/问题1_最终报告_特征提取与时序对齐.md` | 公共时间轴、掩码和特征审计 |
| `q2_baseline/REPORT.md` | 问题2候选模型、baseline和最终比较 |
| `q2_baseline/问题2_附件3测试报告.md` | 附件3统一接口专项推理 |
| `q3_explainability/问题3_最终报告_第二问最优模型解释版.md` | 最终模型解释、附件4预测和证据链 |
| `q3_explainability/q3_aligned_text_bert_final/` | 论文当前采用的最终复杂主模型结果 |
| `q3_explainability/q3_q2_best_run_final/` | 保留的历史/对照解释结果，便于核对旧报告，不作为当前主模型入口 |

## 5. 编译和复现

从仓库根目录执行：

```powershell
Set-Location '2026华为杯研赛latex模板/E题论文'
python make_statistical_figures.py
python make_figures.py
node make_ppt_diagrams.js
node export_diagram_svgs.js
python make_revision_figures.py
python make_explanatory_figures.py
$env:NODE_PATH = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
node make_three_question_roadmap.js
powershell -NoProfile -ExecutionPolicy Bypass -File export_three_question_roadmap.ps1
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf '../../output/pdf/多模态情感识别数学建模论文.pdf' -Force
```

脚本中的 `ROOT = Path(__file__).resolve().parents[2]` 指向仓库根目录，因此 `q2_baseline/`、`q3_explainability/` 和 `emotion-transfer-pilot/` 必须保持在仓库根目录下。

## 6. 本地保留但不上传 Git 的内容

超大原始数据、视频、Pickle 特征和中间工作目录仍保留在本机原路径，用于必要时重跑实验，但已通过 `.gitignore` 排除，避免超过 GitHub 单文件限制。参考论文、临时渲染和非最终测试运行统一放入 `archive/non_final/`，不参与最终论文编译。

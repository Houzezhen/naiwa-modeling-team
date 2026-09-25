# 非数据型图示审查记录

## 图示清单

| 论文图 | 图示作用 | 现有可编辑源 | 论文用图 | 审查结果 |
| --- | --- | --- | --- | --- |
| `fig_pipeline` | 原始媒体、特征、时间轴、掩码与预测证据的路线 | `make_ppt_diagrams.js` 生成的 PPT 对象及 SVG | `fig_pipeline.pdf` | 箭头顺序、分层关系和中文标注清晰，保留 |
| `fig_model_architecture` | 三路时序编码、时间注意力、模态门控与双任务输出 | `make_ppt_diagrams.js` 生成的 PPT 对象及 SVG | `fig_model_architecture.pdf` | 与正文模型结构一致，保留 |

## 未新建 DrawIO 图示的原因

本轮逐一检查了两张已嵌入论文的非数据型图示。现有 PPT 对象可编辑，SVG 和 PDF 均为矢量输出，已覆盖技术路线和模型结构。再生成同内容 DrawIO 图会重复现有图示，不增加论证信息，故没有新增 `fig_roadmap`、各问题流程图或替换当前框图。数据型图表由 Python 脚本处理，不用 DrawIO 重画。

## 自检与嵌入

两张图的节点没有遮挡，箭头流向清楚；正文分别在绪论和问题二模型结构处引用，并用文字说明图中各环节。当前 PDF 使用的框图仍由 `make_ppt_diagrams.js` 与 `export_diagram_svgs.js` 复现。

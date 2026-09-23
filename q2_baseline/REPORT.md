# 第二问模型实验报告

## 1. 评估说明

以下结果均来自固定验证集，不使用测试集选择模型。分类任务同时报告 Accuracy 和 Macro-F1；回归任务报告 MAE 和 Pearson。由于类别可能不均衡，主模型选择优先参考 Macro-F1，并兼顾 MAE、Pearson 和整体 Accuracy。

## 2. Accuracy 大于 60% 的模型

| 模型 | 数据/训练方式 | Accuracy | Macro-F1 | MAE | Pearson | 结论 |
|---|---|---:|---:|---:|---:|---|
| Fixed aligned baseline | `aligned_features.pkl`，原始三模态 baseline | 0.612637 | 0.588115 | **0.616443** | **0.624582** | 回归参考模型 |
| Length-aware resampled baseline | `unaligned_resampled_50.pkl`，长度感知重采样 | 0.614011 | 0.595798 | 0.614132 | 0.620838 | 数据版本候选 |
| Label-free SSL fusion | `unaligned_50.pkl` 自监督 encoder + aligned fusion | 0.608516 | 0.578294 | 0.667391 | 0.573212 | 不选用，表征与情绪任务不匹配 |
| Bidirectional SSL fusion | 双向 GRU + label-free encoder + aligned fusion | 0.618132 | 0.598285 | 0.621818 | 0.588408 | 保留作对比 |
| Residual fusion (`regression_weight=0.5`) | SSL encoder + 零初始化均值/方差残差旁路 | 0.622253 | **0.610330** | 0.623941 | 0.601396 | **主模型** |
| Residual fusion (`regression_weight=1.0`) | 同上，提高回归损失权重 | **0.627747** | 0.605091 | 0.632152 | 0.615692 | Accuracy 对比最优 |

## 3. 主模型选择

最终主模型选择：

```text
q2_multiview_residual_fusion/fusion_best.pt
```

选择理由：

- Macro-F1 达到 `0.610330`，是当前所有实验最高值；
- Accuracy 达到 `0.622253`，明显高于原始 baseline 的 `0.612637`；
- MAE 为 `0.623941`，虽然略高于原始 baseline，但明显优于初始 SSL fusion 的 `0.667391`；
- 验证选择分数为 `0.549990`，高于初始 SSL fusion 的 `0.512469`；
- 相比单纯追求 Accuracy 的 `regression_weight=1.0` 版本，Macro-F1 更高、MAE 更低，综合更平衡。

Accuracy 最高的对比模型单独保留：

```text
q2_multiview_residual_reg1/fusion_best.pt
```

## 4. 模型结构变化

主模型先使用未对齐数据进行无标签表征学习，再使用 aligned 数据进行情绪监督融合。融合阶段在三个 encoder 表征之外增加每个模态的 masked mean/std 残差旁路，残差投影层采用零初始化；encoder 微调学习率降低到 `5e-5`，融合层学习率为 `2e-4`。

```text
三个 SSL encoder 表征
          +
原始 aligned 特征的 masked mean/std 残差
          ↓
多模态融合层
          ↓
分类头 + 回归头
```

## 5. 实验结论

单纯依靠无标签自监督表征直接进行情绪融合效果较差，初始 SSL fusion 的 Macro-F1 和回归指标均下降。加入原始特征残差旁路后，分类效果明显恢复并超过 baseline，说明 SSL 表征可以作为辅助初始化，但不能完全替代情绪相关的原始特征路径。

当前结论是：分类主任务采用 `q2_multiview_residual_fusion/fusion_best.pt`；原始 `q2_baseline_final/best.pt` 继续作为回归和整体稳定性的参考模型；`regression_weight=1.0` 版本作为 Accuracy 最高的对比模型保留。

## 6. 环境备注

服务器日志显示当前 PyTorch 版本不包含 RTX PRO 5000 Blackwell 的 `sm_120` CUDA kernel，仅支持到 `sm_90`。升级到支持该 GPU 的 PyTorch 后，应重新确认最终指标，再进行正式测试集评估。

## 7. 测试集最终评估

测试集只用于最终报告，没有参与模型选择。aligned 数据上的结果如下：

| 模型 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| Fixed aligned baseline | 0.650619 | 0.604127 | 0.657608 | 0.653185 |
| Residual fusion (`regression_weight=0.5`) | 0.628611 | 0.592220 | 0.673816 | 0.634403 |
| Residual fusion (`regression_weight=1.0`) | **0.657497** | **0.612718** | 0.673889 | **0.654312** |

此前的 `unaligned_resampled_50` baseline 在其对应数据版本测试集上的结果为 `Accuracy=0.664374`、`Macro-F1=0.617797`、`MAE=0.649980`、`Pearson=0.655419`。它在本次测试集上四项指标均较强，但由于测试集不能反向用于选模，仍将其作为独立数据版本候选报告，不修改此前基于验证集确定的主模型。

测试集文件和评估记录保存在服务器的 `q2_test_eval/` 与 `q2_test_eval_resampled/` 目录。正式论文应同时报告验证集模型选择结果和测试集最终结果，并说明两者没有混用。

# 第三问：可解释性情感预测建模与验证报告（早期探索版）

> 当前正式结果不使用本文件中的独立注意力模型，而使用第二问最优模型做后处理解释。正式报告见 `REPORT_Q2_BACKBONE.md`，正式产物位于 `q3_q2_best_run_final/`。

## 1. 任务与数据

第三问要求在三模态信息完整的条件下，同时输出情感极性、情感强度、主要参考模态、模态作用程度和关键证据位置。模型参数只使用附件2训练集学习，并使用附件2验证集选择训练轮次；附件4无标签专项测试集只用于最终推理。

本实现使用附件2的 `aligned_50.pkl`。文本、语音、视觉均被组织为50个公共时间窗，输入维度分别为768、74、35。附件4使用对齐版本的20个特征文件及同名视频，窗口映射为：

```text
I_k = [kT/50, (k+1)T/50)
```

## 2. 模型

每个模态使用独立的双向GRU时序编码器，对编码后的50个时间位置计算带掩码的时间注意力 `alpha_(m,k)`。三个模态的时间池化表示输入模态门控网络，得到样本级模态作用程度 `beta_text、beta_audio、beta_vision`。加权后的三模态表示经过融合层，同时输入分类头和回归头。

训练损失为：

```text
L = CrossEntropy + 0.5 * SmoothL1
```

编码器从第二问双向GRU checkpoint 初始化，新增注意力、门控和预测头随机初始化。训练共15轮，按照验证集 `Macro-F1 - 0.1*MAE` 保存最佳模型。

## 3. 解释方法

注意力权重表示模态内部各时间窗的关注程度，模态门控权重表示当前样本中三个模态的相对作用。对注意力最高的窗口做局部遮挡，计算预测类别置信度下降：

```text
drop_(m,k) = p_original - p_masked
Evidence_(m,k) = alpha_(m,k) * beta_m * max(drop_(m,k), 0)
```

验证阶段比较注意力最高窗口和随机窗口的平均置信度下降。附件4结果记录注意力分数、模态贡献、遮挡下降、综合证据分数、时间区间、文本片段和视觉关键帧路径。

## 4. 验证集结果

训练输出保存在 `q3_run/validation.json`，解释验证输出保存在 `q3_run/explanation_validation.json`。

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.631868 |
| Macro-F1 | 0.594578 |
| MAE | 0.603518 |
| Pearson | 0.631473 |
| 注意力最高窗口平均置信度下降 | 0.007674 |
| 随机窗口平均置信度下降 | 0.000964 |
| Top - Random | 0.006710 |

本次最佳模型为第4轮，而不是最后一轮；后续报告应使用 `q3_explainable_best.pt` 的验证结果，避免过拟合轮次污染结论。

## 5. 输出文件

```text
q3_run/q3_explainable_best.pt
q3_run/validation.json
q3_run/explanation_validation.json
q3_run/attachment4_final/q3_predictions.csv
q3_run/attachment4_final/q3_explanations.csv
q3_run/attachment4_final/evidence_frames/
```

文本特征没有逐词真实时间戳，因此文本证据片段按原始转写词序与50个公共窗口做弱映射。当前 Conda 环境的视频解码库与 NumPy 版本存在底层兼容问题，为保证专项推理完整完成，CSV中的 `start_time/end_time` 使用归一化时间位置 `[k/50,(k+1)/50]`，`frame_path` 暂为空；拿到可用视频解码环境后可按同一窗口索引导出实际秒级关键帧。该解释结果是“模型内部重要性 + 遮挡验证”，不宣称严格因果解释。

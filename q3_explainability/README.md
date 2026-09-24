# 问题3可解释性情感预测（统一编码最终模型）

本目录当前正式版本使用统一 `text_bert + aligned_50` 接口训练的复杂多模态模型 `q3_aligned_text_bert_final/q3_explainable_best.pt`。模型包含三路两层双向GRU、时间注意力、模态门控、多层融合和分类/回归双头；通过模态消融和50个时间窗的局部遮挡生成可解释结果，并对附件4输出预测、模态贡献、关键证据和视觉关键帧。旧 `residual_fusion.pt` 仅作为历史指标对照。

## 运行

```powershell
$py = "D:\Anaconda3\envs\pytorch\python.exe"
$data = "..\E题复杂场景下多模态情感识别的数学建模与算法设计\E题数据\E题数据\附件2-数据集特征文件\aligned_50.pkl"
$a4 = "..\E题复杂场景下多模态情感识别的数学建模与算法设计\E题数据\E题数据\附件4-可解释专项视频样本与特征文件\附件4-可解释专项视频样本与特征文件"
$best = "q3_aligned_text_bert_final\q3_explainable_best.pt"
& $py q3_pipeline.py infer-attachment4 --checkpoint $best --attachment4 $a4 --output q3_aligned_text_bert_final\attachment4 --device cuda:0
```

正式版报告见 `问题3_最终报告_第二问最优模型解释版.md`。

本实现兼容本机 `D:\Anaconda3\envs\pytorch`（PyTorch 1.8、NumPy 1.24）读取由较新 NumPy 版本生成的特征文件。如果显存不足，将 batch size 改为16或32。

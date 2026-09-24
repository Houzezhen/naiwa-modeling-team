# 问题3可解释性情感预测

本目录当前正式版本使用第二问验证集最优的 `residual_fusion.pt` 作为预测主干，不重新训练第三问预测模型；通过模态消融和50个时间窗的局部遮挡生成可解释结果，并对附件4输出预测、模态贡献、关键证据和视觉关键帧。正式结果以 `q2_posthoc_explain.py` 为准。

## 运行

```powershell
$py = "D:\Anaconda3\envs\pytorch\python.exe"
$data = "..\E题复杂场景下多模态情感识别的数学建模与算法设计\E题数据\E题数据\附件2-数据集特征文件\aligned_50.pkl"
$a4 = "..\E题复杂场景下多模态情感识别的数学建模与算法设计\E题数据\E题数据\附件4-可解释专项视频样本与特征文件\附件4-可解释专项视频样本与特征文件"
$best = "..\q2_baseline\checkpoints\residual_fusion.pt"
& $py q2_posthoc_explain.py --checkpoint $best --data $data --attachment4 $a4 --output q3_q2_best_run_final --device cuda:0
```

正式版报告见 `问题3_最终报告_第二问最优模型解释版.md`。

本实现兼容本机 `D:\Anaconda3\envs\pytorch`（PyTorch 1.8、NumPy 1.24）读取由较新 NumPy 版本生成的特征文件。如果显存不足，将 batch size 改为16或32。

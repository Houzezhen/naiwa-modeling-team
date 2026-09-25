"""问题2：局部时段模态缺失下的鲁棒情感预测（DM-CMRN 工程包）。

模块划分：
    src.paths    本地数据与输出路径
    src.data     数据加载、双掩码（padding / missing）、缺失模拟、标准化
    src.models   编码器、4 个 Baseline、DM-CMRN（双掩码 + 局部连续重建 + 质量门控）
    src.losses   分类 / 回归 / 重建 / 蒸馏损失
    src.engine   训练与评估循环
入口脚本：
    tools/audit_data.py       数据审计（区分 padding 零与有效长度内的缺失零）
    train.py                  训练（baseline / teacher / student 两阶段）
    evaluate.py               验证集与 test 划分评估（含场景化缺失）
    missing_analysis.py       缺失类型/位置/时长/缺失率网格分析
    infer_attachment3.py      附件3 推理（自动检测连续缺失区间）
    tools/run_all.py          一键按推荐顺序串跑
"""

__all__ = ["paths", "data", "models", "losses", "engine"]

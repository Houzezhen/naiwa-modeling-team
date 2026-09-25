"""本地路径常量（与 q2_baseline、problem2 的既有约定保持一致；换机器只需改这里）。"""

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # q2_dm_cmrn/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # naiwa-modeling-team/

DATA_ROOT = Path(r"E:\研究生数学建模\赛题\E题复杂场景下多模态情感识别的数学建模与算法设计\E题数据")
FEATURE_DIR = DATA_ROOT / "附件2-数据集特征文件"
ALIGNED_PKL = FEATURE_DIR / "aligned_50.pkl"                  # text(50,768)+audio(50,74)+vision(50,35)
RESAMPLED_PKL = FEATURE_DIR / "unaligned_resampled_50.pkl"    # 长度感知重采样版（可选数据版本）
LABEL_XLSX = FEATURE_DIR / "label.xlsx"
ATTACHMENT3_DIR = DATA_ROOT / "附件3-模态缺失特征样本"
ATTACHMENT1_DIR = DATA_ROOT / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"

DATA_KINDS = {
    "aligned": ALIGNED_PKL,
    "resampled": RESAMPLED_PKL,
}
OUTPUT_ROOT = HERE / "outputs"

MODALITIES = ("text", "audio", "vision")
INPUT_DIMS = {"text": 768, "audio": 74, "vision": 35}
CLASS_NAMES = ("Negative", "Neutral", "Positive")

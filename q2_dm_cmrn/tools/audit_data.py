"""数据审计：区分“长度填充零”与“有效段内的缺失零”，为双掩码设计提供证据。

运行：
    python tools/audit_data.py                 # 审计 aligned + resampled
    python tools/audit_data.py --data-kind aligned
产出：outputs/audit/audit_<kind>.json 与终端摘要。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.data import (build_exist_masks, mask_intervals, row_nonzero,  # noqa: E402
                      span_from_nonzero)
from src.paths import DATA_KINDS, MODALITIES, OUTPUT_ROOT  # noqa: E402


def audit_split(split: dict, split_name: str) -> dict:
    exist = build_exist_masks(split)
    report = {"split": split_name, "samples": len(split["id"]), "modalities": {}}
    for name in MODALITIES:
        values = np.asarray(split[name])
        observed = exist[name] & row_nonzero(values)
        padding = ~exist[name]
        missing = exist[name] & ~observed
        counts = exist[name].sum(axis=1)
        report["modalities"][name] = {
            "shape": list(values.shape), "dtype": str(values.dtype),
            "exist_ratio": float(exist[name].mean()), "observed_ratio": float(observed.mean()),
            "padding_ratio": float(padding.mean()), "interior_missing_ratio": float(missing.mean()),
            "interior_missing_bins": int(missing.sum()),
            "mean_exist_bins": float(counts.mean()),
            "contiguous_exist_share": float(np.mean([len(mask_intervals(row)) <= 1 for row in exist[name]])),
            "first_position_exist_share": float(np.mean(exist[name][:, 0])),
        }
    report["labels"] = {"class_counts": np.bincount(
        np.asarray(split["classification_labels"], dtype=np.int64), minlength=3).tolist(),
        "regression_min": float(np.min(split["regression_labels"])),
        "regression_max": float(np.max(split["regression_labels"]))}
    report["has_explicit_masks"] = [name for name in MODALITIES if f"{name}_mask" in split]
    report["has_lengths"] = [name for name in MODALITIES if f"{name}_lengths" in split]
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-kind", choices=tuple(DATA_KINDS), default=None,
                        help="默认审计 aligned 与 resampled 两个数据版本")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "audit")
    args = parser.parse_args(argv)

    kinds = [args.data_kind] if args.data_kind else list(DATA_KINDS)
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        path = DATA_KINDS[kind]
        if not path.is_file():
            print(f"跳过（文件不存在）：{path}", flush=True)
            continue
        print(f"\n=== 审计 {kind}：{path} ===", flush=True)
        with path.open("rb") as handle:
            import pickle
            source = pickle.load(handle)
        payload = {"data_kind": kind, "path": str(path), "splits": {}}
        for split_name in list(source.keys()):
            split = source.pop(split_name)
            report = audit_split(split, split_name)
            payload["splits"][split_name] = report
            print(f"[{split_name}] {report['samples']} 条；"
                  + " ".join(f"{name}: exist={meta['exist_ratio']:.3f} "
                             f"observ={meta['observed_ratio']:.3f} "
                             f"interior_missing={meta['interior_missing_bins']}"
                             for name, meta in report["modalities"].items()), flush=True)
            del split
        (args.output / f"audit_{kind}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        del source
    print(f"\n已写入：{args.output}", flush=True)


if __name__ == "__main__":
    main()

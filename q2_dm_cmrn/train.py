"""训练入口：支持 Baseline 单阶段、Teacher（完整模态）与 Student（DM-CMRN + 局部重建 + 蒸馏）。

示例：
    python train.py --model cross_attn --epochs 8 --output outputs/cross_attn
    python train.py --model cross_attn --stage teacher --clean-only --output outputs/teacher
    python train.py --model dm_cmrn --stage student --teacher outputs/teacher/best.pt \
        --rec-weight 0.3 --distill-weight 1.0 --output outputs/dm_cmrn
    python train.py --model dm_cmrn --variant dm_cmrn_no_lcr --rec-weight 0 --output outputs/abl_no_lcr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.data import Scenario  # noqa: E402
from src.engine import (load_checkpoint, load_prepared_splits, make_loader, resolve_device,  # noqa: E402
                        run_epoch, save_checkpoint, set_seed)
from src.models import ABLATION_VARIANTS, build_model  # noqa: E402
from src.paths import DATA_KINDS, OUTPUT_ROOT  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="默认取 src.paths 中的 aligned 文件")
    parser.add_argument("--data-kind", choices=tuple(DATA_KINDS), default="aligned")
    parser.add_argument("--model", choices=("early", "cross_attn", "misa", "dm_cmrn"), default="dm_cmrn")
    parser.add_argument("--variant", choices=tuple(ABLATION_VARIANTS), default=None,
                        help="DM-CMRN 消融变体")
    parser.add_argument("--stage", choices=("single", "teacher", "student"), default="single")
    parser.add_argument("--teacher", type=Path, default=None, help="student 阶段的教师检查点")
    parser.add_argument("--warm-start-encoders", action="store_true",
                        help="用教师的模态编码器权重热启动学生编码器（同为 ModalEncoder，仅状态嵌入缺失）")
    parser.add_argument("--clean-only", action="store_true", help="训练时不注入缺失（Teacher 用）")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ctx-span", type=int, default=6)
    parser.add_argument("--reg-weight", type=float, default=0.5)
    parser.add_argument("--rec-weight", type=float, default=0.3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--similarity-weight", type=float, default=0.0, help="MISA 表示约束权重")
    parser.add_argument("--rec-target", choices=("self", "teacher", "context"), default=None)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int, default=0, help="冒烟测试：只用前 N 条训练")
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data_path = args.data or DATA_KINDS[args.data_kind]
    run_dir = args.output or (OUTPUT_ROOT / (args.variant or f"{args.model}_{args.stage}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model_kwargs = {"hidden": args.hidden, "layers": args.layers, "nhead": args.nhead,
                    "dropout": args.dropout, "ctx_span": args.ctx_span}
    if args.variant:
        model_kwargs.update(ABLATION_VARIANTS[args.variant])
    model = build_model(args.model, **model_kwargs).to(device)
    rec_target = args.rec_target or ("teacher" if args.teacher else "self")
    print(f"device={device} model={args.model} variant={args.variant} stage={args.stage} "
          f"rec_target={rec_target} params={sum(p.numel() for p in model.parameters())}", flush=True)

    print(f"加载数据：{data_path}", flush=True)
    bundle = load_prepared_splits(Path(data_path), need_test=False, max_train=args.max_train,
                                  max_valid=args.max_valid)
    prepared, normalizers = bundle["prepared"], bundle["normalizers"]
    counts = np.bincount(prepared["train"]["class"], minlength=3).astype(np.float32)
    class_weights = None if args.no_class_weights else torch.tensor(
        counts.sum() / np.maximum(counts, 1) / 3.0, dtype=torch.float32, device=device)

    clean_scenario = Scenario(mode="none")
    missing_scenario = Scenario(mode="random", position="random", ratio=0.3)
    train_dataset, train_loader = make_loader(prepared["train"], args.batch_size, args.seed,
                                              train_mode=not args.clean_only, with_clean=True, shuffle=True)
    _, valid_clean_loader = make_loader(prepared["valid"], args.batch_size, args.seed, train_mode=False,
                                        scenario=clean_scenario, with_clean=False)
    _, valid_missing_loader = make_loader(prepared["valid"], args.batch_size, args.seed, train_mode=False,
                                          scenario=missing_scenario, with_clean=False)

    teacher = None
    if args.stage == "student" and args.teacher:
        teacher, payload = load_checkpoint(Path(args.teacher), device)
        print(f"已加载教师：{args.teacher}", flush=True)
        if args.warm_start_encoders and hasattr(teacher, "encoders") and hasattr(model, "encoders"):
            missing, unexpected = model.encoders.load_state_dict(teacher.encoders.state_dict(), strict=False)
            print(f"已用教师编码器热启动学生（未匹配键 missing={list(missing)} "
                  f"unexpected={list(unexpected)}）", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    settings = {"reg_weight": args.reg_weight,
                "rec_weight": args.rec_weight if args.model == "dm_cmrn" else 0.0,
                "distill_weight": args.distill_weight if teacher is not None else 0.0,
                "similarity_weight": args.similarity_weight, "class_weights": class_weights,
                "rec_target": rec_target, "ctx_span": args.ctx_span, "temperature": args.temperature,
                "clip": 1.0}

    history, best_score, best_epoch, stale = [], -np.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_dataset.reseed(epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, device, settings, teacher=teacher, train=True)
        clean_metrics = run_epoch(model, valid_clean_loader, None, device, settings, train=False)
        missing_metrics = run_epoch(model, valid_missing_loader, None, device, settings, train=False)
        score = 0.5 * (clean_metrics["macro_f1"] + missing_metrics["macro_f1"]) + 0.1 * missing_metrics["pearson"]
        scheduler.step(score)
        row = {"epoch": epoch, "score": score, "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.time() - started, 1)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()
                    if key != "prediction_distribution"})
        row.update({f"valid_clean_{key}": value for key, value in clean_metrics.items()
                    if key != "prediction_distribution"})
        row.update({f"valid_missing_{key}": value for key, value in missing_metrics.items()
                    if key != "prediction_distribution"})
        history.append(row)
        print(f"epoch {epoch:03d} loss={train_metrics['loss']:.4f} "
              f"clean[f1={clean_metrics['macro_f1']:.4f} acc={clean_metrics['accuracy']:.4f}] "
              f"missing[f1={missing_metrics['macro_f1']:.4f} acc={missing_metrics['accuracy']:.4f} "
              f"mae={missing_metrics['mae']:.4f} r={missing_metrics['pearson']:.4f}] "
              f"score={score:.4f} ({row['seconds']:.0f}s)", flush=True)
        if score > best_score + args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, args.model, model_kwargs, normalizers,
                            {"valid_clean": clean_metrics, "valid_missing": missing_metrics,
                             "score": score, "epoch": epoch}, extra={"training_args": vars(args)})
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}（best epoch={best_epoch}, score={best_score:.4f}）",
                      flush=True)
                break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    (run_dir / "config.json").write_text(json.dumps(
        {"training_args": {key: str(value) for key, value in vars(args).items()} | {"data": str(data_path)},
         "model_kwargs": model_kwargs,
         "settings": {key: (value.tolist() if torch.is_tensor(value) else value)
                      for key, value in settings.items()},
         "best_epoch": best_epoch, "best_score": best_score}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"完成：best epoch={best_epoch}，检查点 {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()

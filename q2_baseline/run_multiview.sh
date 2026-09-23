#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"
DEVICE="${DEVICE:-cuda:0}"

if [[ "$MODE" != "separate" && "$MODE" != "hybrid" ]]; then
    echo "usage: CONDA_ENV=环境名 bash run_multiview.sh separate|hybrid"
    exit 2
fi

SESSION="q2_multiview_${MODE}"
LOG="$ROOT/q2_multiview_${MODE}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION"
    echo "tmux attach -t $SESSION"
    exit 1
fi

if [[ "$MODE" == "separate" ]]; then
    OUTPUT="$ROOT/q2_multiview_separate"
    TRAIN_ARGS=(
        --mode separate
        --aligned-data "$ROOT/aligned_features.pkl"
        --unaligned-data "$ROOT/unaligned_50.pkl"
        --output "$OUTPUT"
        --hidden 128
        --layers 2
        --projector-hidden 128
        --batch-size 64
        --encoder-epochs 10
        --fusion-epochs 20
        --freeze-epochs 3
        --encoder-learning-rate 3e-4
        --fusion-learning-rate 2e-4
        --feature-noise 0.05
        --time-mask-probability 0.15
        --variance-weight 1.0
        --covariance-weight 0.04
        --device "$DEVICE"
        --amp
    )
else
    OUTPUT="$ROOT/q2_multiview_hybrid"
    TRAIN_ARGS=(
        --mode hybrid
        --aligned-data "$ROOT/aligned_features.pkl"
        --unaligned-data "$ROOT/unaligned_50.pkl"
        --init-dir "$ROOT/q2_multiview_separate"
        --output "$OUTPUT"
        --hidden 128
        --layers 2
        --hybrid-batch-size 32
        --hybrid-epochs 15
        --hybrid-learning-rate 1e-4
        --consistency-weight 0.25
        --logit-consistency-weight 0.5
        --device "$DEVICE"
        --amp
    )
fi

printf -v TRAIN_CMD '%q ' python -u "$ROOT/train_multiview.py" "${TRAIN_ARGS[@]}"
RUN="cd $(printf '%q' "$ROOT") && source $(printf '%q' "$CONDA_ROOT/etc/profile.d/conda.sh") && conda activate $(printf '%q' "$CONDA_ENV") && exec $TRAIN_CMD > $(printf '%q' "$LOG") 2>&1"
tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$RUN")"

echo "started: $SESSION"
echo "log: $LOG"
echo "watch: tail -f $LOG"

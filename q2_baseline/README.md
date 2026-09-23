# Question 2 Experiments

## Fixed baseline

Naming note: local `emotion-transfer-pilot/aligned_50.pkl` and server `emotion-transfer-pilot/aligned_features.pkl` are byte-identical copies. The original baseline used the server filename; they are not two different data versions.

`train.py` uses three independent lightweight temporal encoders, masked temporal means, concatenation fusion, and separate classification/regression outputs. The fixed comparison checkpoint is `q2_baseline_final/best.pt` on the server. It uses the original `aligned_features.pkl` and has validation Accuracy `0.6126`, Macro-F1 `0.5881`, MAE `0.6164`, and Pearson `0.6246`.

Keep this checkpoint unchanged when comparing data or architecture changes. The supplied test split and any unlabeled specialty data are not used for selection.

## Cleaned experiments

Failed attention, Transformer, robust-training, sparse-history, and frozen-JEPA diagnostics were removed from the working tree and their remote output directories were cleaned. Their conclusions were recorded before cleanup: none passed the predefined improvement guard. The fixed baseline remains the only original aligned-data model.

## Unaligned feature conversion

`emotion-transfer-pilot/unaligned_50.pkl` contains text at 50 positions, but audio and vision at up to 500 padded positions with per-sample lengths. `prepare_unaligned_50.py` converts audio and vision to 50 equal-duration weighted bins, writes explicit `audio_mask` and `vision_mask`, preserves split boundaries and IDs, and emits float32 arrays. `train.py` prefers explicit masks when available.

```bash
python q2_baseline/prepare_unaligned_50.py \
  --input emotion-transfer-pilot/unaligned_50.pkl \
  --output emotion-transfer-pilot/unaligned_resampled_50.pkl \
  --bins 50
```

The converted file has shapes `(3395,50,74)/(3395,50,35)`, `(728,50,74)/(728,50,35)`, and `(727,50,74)/(727,50,35)` for train/valid/test. It does not overwrite `aligned_50.pkl`.

## Unaligned-data candidate

The same five-epoch baseline configuration was trained on `unaligned_resampled_50.pkl`. Results on the fixed validation split:

| Data version | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|
| Fixed aligned baseline | 0.588115 | 0.616443 | 0.624582 |
| Length-aware resampled | 0.595798 | 0.614132 | 0.620838 |

On the same 27 missing-interval validation scenarios, the resampled candidate changes mean Macro-F1 from `0.577816` to `0.586305` and mean MAE from `0.624742` to `0.622080`. This is a promising candidate, not a replacement for the fixed baseline until independent test evaluation is completed.

Server artifacts:

- `/home/user/code_1/naiwa-modeling-team/emotion-transfer-pilot/unaligned_50.pkl`
- `/home/user/code_1/naiwa-modeling-team/emotion-transfer-pilot/unaligned_resampled_50.pkl`
- `/home/user/code_1/naiwa-modeling-team/emotion-transfer-pilot/q2_unaligned_baseline/best.pt`
- `/home/user/code_1/naiwa-modeling-team/emotion-transfer-pilot/q2_unaligned_baseline/validation_scenarios.json`

## Experiment Log — 2026-09-23

### Objective

Evaluate whether the uploaded variable-length, unaligned audio/vision features can replace the current aligned feature version without changing the baseline architecture.

### Data audit

- Source: `unaligned_50.pkl`, approximately `2.90 GB`.
- Splits: train/valid/test sizes are `3395/728/727`, matching the aligned version.
- Text shape: `(N,50,768)`.
- Audio shape: `(N,500,74)` with `audio_lengths`.
- Vision shape: `(N,500,35)` with `vision_lengths`.
- IDs and labels match the aligned file in the same order.
- The original aligned file is not overwritten.

### Processing

`prepare_unaligned_50.py` trims each audio/vision sequence to its recorded length, divides the valid duration into 50 equal-duration bins, and computes overlap-weighted feature averages. It also writes explicit `audio_mask` and `vision_mask` fields. The resulting `unaligned_resampled_50.pkl` is approximately `848 MB` and has the expected 50-step shapes.

### Training configuration

- Model: unchanged three-modal baseline in `train.py`.
- Epochs: `5`.
- Batch size: `64`.
- Hidden size: `64`.
- Dropout: `0.2`.
- Learning rate: `0.001`.
- Device: server `cuda:0` (RTX 5880 Ada in the installed PyTorch environment).
- Selection: validation split only; test split is not used for model selection.

### Results

| Model/data version | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| Fixed aligned baseline | 0.612637 | 0.588115 | 0.616443 | 0.624582 |
| Length-aware resampled features | 0.614011 | 0.595798 | 0.614132 | 0.620838 |

The resampled version improves validation Macro-F1 by `0.007683` and reduces MAE by `0.002312`; Accuracy increases by `0.001374`. On the predefined 27 missing-interval validation scenarios, mean Macro-F1 changes from `0.577816` to `0.586305`, and mean MAE changes from `0.624742` to `0.622080`.

### Decision

The length-aware resampled data version is retained as a candidate because it improves both clean Macro-F1 and missing-interval performance under the same baseline configuration. The original aligned checkpoint remains the fixed reference and is not overwritten. Final test-set or specialty-set inference should be run only after the candidate/reference choice is documented; no test metric is used in this experiment log.

### Cleanup

Old failed experiment scripts and remote diagnostic directories were removed. Retained remote model artifacts are the fixed `q2_baseline_final/best.pt` and the candidate `q2_unaligned_baseline/best.pt`; diagnostic JSON and validation predictions remain next to the candidate for audit.

## Multi-view training plan

`train_multiview.py` implements two new experiments without mixing test data into selection.

The temporal encoder supports an optional bidirectional GRU. Enable it with `--bidirectional`; the
per-direction hidden width is halved so the pooled representation remains `--hidden`-dimensional.
This creates a separate checkpoint family, so use the flag consistently for both `separate` and
`hybrid` runs and do not reuse unidirectional encoder checkpoints.

### Separate unaligned encoders, aligned fusion

`--mode separate` first trains independent two-layer GRU encoders for text, audio, and vision using `unaligned_50.pkl`. This stage is label-free: each sample produces two randomly masked/noised temporal views, and the shared encoder is trained with an invariance loss plus variance-floor and covariance-decorrelation regularization. Classification and regression labels are not used for optimization or checkpoint selection. Only the encoder state is saved; the temporary projector is discarded. Audio and vision use their recorded lengths, so the networks consume up to 500 time positions.

The three label-free encoder checkpoints are then loaded into a fusion model and trained on `aligned_features.pkl`, the server copy of local `aligned_50.pkl`. Emotion classification and regression are learned only in this fusion stage. The fusion stage freezes encoders first, then unfreezes them with the lower fusion learning rate.

```bash
cd ~/code_1/naiwa-modeling-team/emotion-transfer-pilot
tmux new-session -d -s q2_multiview_separate "~/miniforge3/bin/python -u train_multiview.py --mode separate --aligned-data aligned_features.pkl --unaligned-data unaligned_50.pkl --output q2_multiview_separate --hidden 128 --layers 2 --batch-size 64 --encoder-epochs 10 --fusion-epochs 20 --freeze-epochs 3 --encoder-learning-rate 3e-4 --fusion-learning-rate 2e-4 --bidirectional --device cuda:0 --amp > q2_multiview_separate.log 2>&1"
tail -f q2_multiview_separate.log
```

### Hybrid two-view training

Run this only after the separate experiment finishes. `--mode hybrid` initializes from the label-free encoders and jointly trains on the unaligned and aligned views of the same samples. This is a supervised final-stage experiment: it uses emotion losses on both views plus representation, classification-logit, and regression consistency losses. Validation and missing-interval reports use the aligned view.

```bash
cd ~/code_1/naiwa-modeling-team/emotion-transfer-pilot
tmux new-session -d -s q2_multiview_hybrid "~/miniforge3/bin/python -u train_multiview.py --mode hybrid --aligned-data aligned_features.pkl --unaligned-data unaligned_50.pkl --init-dir q2_multiview_separate --output q2_multiview_hybrid --hidden 128 --layers 2 --hybrid-batch-size 32 --hybrid-epochs 15 --hybrid-learning-rate 1e-4 --consistency-weight 0.25 --logit-consistency-weight 0.5 --bidirectional --device cuda:0 --amp > q2_multiview_hybrid.log 2>&1"
tail -f q2_multiview_hybrid.log
```

The defaults use more GPU memory than the old 64-dimensional CNN baseline while keeping the 500-step GRU batch conservative. If CUDA runs out of memory, lower `--batch-size` to `32` or `--hybrid-batch-size` to `16`. Expected outputs are `q2_multiview_separate/encoders/`, `fusion_best.pt`, `fusion_validation.json`, and `q2_multiview_hybrid/hybrid_best.pt`, `hybrid_validation.json`.

### Residual fusion follow-up

The first label-free encoder plus fusion run underperformed the fixed baseline (`f1_macro=0.578294`, `mae=0.667391`, `pearson=0.573212`). A follow-up was run without retraining the encoders: the fusion model received a zero-initialized residual path from each modality's masked mean and standard deviation, and encoder fine-tuning used `5e-5` instead of the fusion learning rate.

| Model | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| Fixed aligned baseline | 0.612637 | 0.588115 | 0.616443 | 0.624582 |
| Residual fusion, regression weight 0.5 | 0.622253 | 0.610330 | 0.623941 | 0.601396 |
| Residual fusion, regression weight 1.0 | 0.627747 | 0.605091 | 0.632152 | 0.615692 |

The residual fusion checkpoint is better for classification, while the fixed baseline remains better for regression MAE and Pearson. The server also reports that the installed PyTorch build does not contain kernels for the RTX PRO 5000 Blackwell (`sm_120`); the environment should be upgraded before treating small metric differences as final conclusions.

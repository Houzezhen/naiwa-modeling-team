# Question 2 Checkpoints

These are the selected validation-trained checkpoints used for the final test evaluation. They are ordinary Git files; Git LFS is not required because each file is below the GitHub single-file limit.

| File | Source experiment | Test Accuracy | Test Macro-F1 | Test MAE | Test Pearson |
|---|---|---:|---:|---:|---:|
| `aligned_baseline.pt` | Original aligned baseline | 0.650619 | 0.604127 | 0.657608 | 0.653185 |
| `unaligned_resampled_baseline.pt` | Length-aware resampled baseline | 0.664374 | 0.617797 | 0.649980 | 0.655419 |
| `residual_fusion.pt` | SSL encoder plus residual fusion, regression weight 0.5 | 0.628611 | 0.592220 | 0.673816 | 0.634403 |
| `residual_fusion_reg1.pt` | SSL encoder plus residual fusion, regression weight 1.0 | 0.657497 | 0.612718 | 0.673889 | 0.654312 |

The validation-selected balanced main model is `residual_fusion.pt`. The strongest final test candidate is `unaligned_resampled_baseline.pt`; it was not selected using test results and is retained as a separate data-version comparison. `residual_fusion_reg1.pt` is retained because it is the strongest aligned residual-fusion variant on the test set.

SHA256 values are recorded in `SHA256SUMS.txt`.

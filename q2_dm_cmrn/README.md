# 问题2：局部时段模态缺失下的鲁棒情感预测（DM-CMRN）

本目录是针对“**局部时段模态缺失**”（不是整模态完全缺失）这一赛题设定的完整工程实现：先建立可靠
Baseline 群，再引入三大针对局部缺失的创新点，并配套缺失规律分析与附件3 推理。

```
q2_dm_cmrn/
├── src/
│   ├── paths.py      本地数据/输出路径常量（换机器只需改这里）
│   ├── data.py       双掩码构建、缺失模拟、标准化、Dataset
│   ├── models.py     模态编码器、4 个 Baseline、DM-CMRN（三大创新）
│   ├── losses.py     分类/回归/重建/蒸馏/表示约束损失
│   └── engine.py     设备解析、数据准备、指标（含 CCC）、训练与评估循环、检查点
├── tools/
│   ├── audit_data.py 数据审计：区分长度填充零与有效段内缺失零
│   └── run_all.py    一键串跑全部步骤（支持 --smoke / --dry-run）
├── train.py          训练入口（single / teacher / student 三阶段）
├── evaluate.py       多缺失场景评估（干净、单/双/三模态缺失）
├── missing_analysis.py  缺失类型/位置/时长/缺失率网格分析
├── infer_attachment3.py 附件3 推理（自动检测连续缺失区间）
└── outputs/          运行产物
```

## 1 核心设计

### 1.1 双掩码（创新点 1）

| 掩码 | 含义 | 来源 |
|---|---|---|
| `exist` | 位于该样本有效时间轴内（长度边界内） | 显式 `*_mask` / `*_lengths`，否则由非零段推断 |
| `observed` | 有效轴内且确实有观测（特征非全零） | 由特征本身判定 |
| `missing` | 有效轴内本该存在但缺失 | `exist & ~observed` |

- 编码器自注意力只对 `observed` 位置开放：`missing`（值不可靠）与 `padding`（本来就不存在）都被屏蔽；
- 同时用**三值状态嵌入**（0=padding / 1=observed / 2=missing）显式告诉模型“这里本该有但丢了”；
- 这样模型不会把“局部缺失的零”和“长度填充的零”混为一谈（消融 `dm_cmrn_no_dual_mask` 专门验证其价值）。

> 数据侧事实：附件2 对齐版/重采样版的有效段内部**没有**零，零值几乎全部来自长度填充（可用
> `tools/audit_data.py` 复现；附件3 才是真实局部缺失）。因此训练时必须**人为制造连续缺失**，
> 推理时用同样的“有效段内全零”判据检测缺失，保证训练/推理语义一致。

### 1.2 局部连续重建（创新点 2，`models.LocalContinuousReconstruction`）

- 对每个缺失位置，只在**前后各 `ctx_span` 个时间位置**内、且**有观测**的 token 上做跨模态注意力：
  目标模态的邻域上下文 + 另外两个模态对应时段的观测 token；
- 查询向量 = 时间位置嵌入 + 模态嵌入（缺失位置原始值为零，不可作 query 值）；
- 重建损失只统计“有缺失且重建有效”的位置：`L_rec = mean_{t∈M} ||f̂_t − f_t^target||²`，
  目标可选 `self`（干净输入自编码）/ `teacher`（完整模态教师）/ `context`（邻域观测均值）；
- `rec_mode="global"` 时窗口不受限，得到“全局重建”对照，用于验证“局部”这一结构先验的价值。

### 1.3 质量门控融合（创新点 3，`models.QualityAwareGate`）

`g_m = sigmoid(W_g [h_m; r_m])`，其中 `r_m = [观测比例, 有效轴比例, 缺失比例, 重建置信度]`；
缺失模态在 softmax 前被 `-1e4` 屏蔽，权重自动流向可靠模态。消融 `dm_cmrn_no_qagf` 用
“按观测比例朴素加权”替代。

### 1.4 训练策略

- 三阶段：`--stage single`（Baseline）/ `--stage teacher --clean-only`（完整模态教师）/
  `--stage dm_cmrn student --teacher ...`（随机缺失 + 局部重建 + 蒸馏）；
- 缺失模拟：在有效段内随机 1–3 个连续区间，缺失率 10%–50%，模态数 1–3 个，位置随起/中/末/随机；
  每个 epoch 重新采样（`MultimodalDataset.reseed`）；
- 总损失：`L = L_cls + λ1·L_reg + λ2·L_rec + λ3·L_distill`（蒸馏 = 温度 KL + 回归 MSE）；
- 选优准则：`0.5×(干净 Macro-F1 + 缺失 Macro-F1) + 0.1×缺失 Pearson`，带 `--min-delta` 抑制噪声；
- 回归头为无界线性输出（不加 tanh），避免强度预测向 0 收缩。

## 2 Baseline 群（`models.py`）

| 模型 | 名称 | 结构要点 |
|---|---|---|
| Baseline 1 | `--model early` | 三模态线性投影 → 掩码均值池化 → 拼接 → MLP（性能下限） |
| Baseline 2 | `--model cross_attn` | 模态内 Transformer + 跨模态注意力（简化 MulT），同时充当蒸馏教师 |
| Baseline 3 | `--model misa` | 不变表示 + 特异表示分解；缺失模态特异表示置零（`--similarity-weight` 可加正交约束） |
| Baseline 4 | `--stage teacher/student` | 完整模态教师 + 学生蒸馏（配合 DM-CMRN 使用） |
| 主模型 | `--model dm_cmrn` | 双掩码 + 局部连续重建 + 质量门控 |

## 3 环境与数据

- 解释器：`D:\Anaconda\envs\Lecture2env\python.exe`（torch 2.6.0+cu124、numpy 1.26.4、transformers 5.17.0）
- 数据（`src/paths.py` 已内置本地绝对路径）：
  - `E题数据\附件2-数据集特征文件\aligned_50.pkl`（默认：text(50,768)+audio(50,74)+vision(50,35)）
  - `E题数据\附件2-数据集特征文件\unaligned_resampled_50.pkl`（`--data-kind resampled`）
  - `E题数据\附件3-模态缺失特征样本\{对齐版本,未对齐版本}`
- 附件3 的 `distilbert` 文本模式需要联网取权重；本机已设 `HF_ENDPOINT=https://hf-mirror.com`。
- 内存：单个数据版本常驻约 0.9 GB，训练峰值约 1.5–2 GB，建议保证 ≥3 GB 可用内存。

## 4 运行顺序（推荐）

```powershell
cd "E:\研究生数学建模\e题代码\naiwa-modeling-team\q2_dm_cmrn"

# 0) 数据审计（确认 padding 与 missing 的区别）
python -u tools\audit_data.py

# 1) Baseline 群
python -u train.py --model early      --epochs 10 --device auto --output outputs\early
python -u train.py --model cross_attn --epochs 10 --device auto --output outputs\cross_attn
python -u train.py --model misa       --epochs 10 --device auto --output outputs\misa

# 2) Teacher（完整模态）
python -u train.py --model cross_attn --stage teacher --clean-only --epochs 10 --device auto --output outputs\teacher

# 3) 主模型 DM-CMRN（随机缺失 + 局部重建 + 蒸馏）
python -u train.py --model dm_cmrn --stage student --teacher outputs\teacher\best.pt `
    --rec-weight 0.3 --distill-weight 1.0 --epochs 12 --device auto --output outputs\dm_cmrn

# 4) 测试划分评估 / 5) 缺失规律 / 6) 附件3 推理
python -u evaluate.py --checkpoint outputs\dm_cmrn\best.pt --split test --device auto
python -u missing_analysis.py --checkpoint outputs\dm_cmrn\best.pt --split valid --device auto
python -u infer_attachment3.py --checkpoint outputs\dm_cmrn\best.pt --device cpu
```

一键等价写法（含 5 个消融）：

```powershell
python -u tools\run_all.py --dry-run     # 先看命令
python -u tools\run_all.py --smoke       # 小样本 1 轮，验证链路
python -u tools\run_all.py               # 正式全流程
```

`run_all.py` 的行为约定：

- 步骤顺序固定为 `audit → baselines → teacher → main → ablations → evaluate → missing → attachment3`，
  可用 `--steps` 选择子集（例如 `--steps audit baselines teacher main`）；
- **`--smoke` 会自动把所有产物隔离到 `outputs\_smoke\`**（可用 `--output-root` 指定其他目录），
  因此冒烟测试不会覆盖正式训练结果；不带 `--smoke` 时写回 `outputs\`；
- `evaluate` / `missing` / `attachment3` 需要读取主模型检查点（`<root>\dm_cmrn\best.pt`），
  若该文件不存在会**自动跳过并提示**（不中断整体流程），此时请先运行 `main`（或 `teacher main`）；
- 某一步失败只会记录并在结尾汇总，不会阻断后续步骤；全部成功时返回码为 0。

## 5 消融实验（`--variant`，见 `models.ABLATION_VARIANTS`）

| 变体 | 含义 | 验证假设 |
|---|---|---|
| `dm_cmrn_full` | 完整模型 | 主结果 |
| `dm_cmrn_no_dual_mask` | 只用 padding mask（缺失与填充不分） | 双掩码的价值 |
| `dm_cmrn_no_lcr` | 去掉局部重建 | LCR 的贡献 |
| `dm_cmrn_global_rec` | 全局重建替代局部重建 | “局部连续”先验的价值 |
| `dm_cmrn_no_qagf` | 朴素观测比例加权替代质量门控 | 质量门控的价值 |

## 5.0 多种子复算与调参结果（正式结论，5 种子 / 3 种子）

运行器：`python -u tools\study_runner.py --study seeds --seeds 42 43 44 45 46`
和 `... --study tune --seeds 42 43 44 --min-delta 0 --patience 6 --epochs 12`；
产物：`outputs\_study\seeds\summary.md`、`outputs\_study\tune\summary.md`（含 metrics.csv / summary.json）。

### (1) `dm_cmrn` vs `cross_attn`：5 个种子，test 划分（均值 ± 标准差）

| 配置 | clean Acc | clean Macro-F1 | clean MAE | clean Pearson | clean CCC | triple_30 Macro-F1 | triple_30 MAE |
|---|---|---|---|---|---|---|---|
| `cross_attn` | 0.6597 ± 0.0176 | **0.6317 ± 0.0177** | 0.6409 ± 0.0126 | 0.6841 ± 0.0201 | 0.6488 ± 0.0405 | **0.6205 ± 0.0168** | 0.6553 ± 0.0142 |
| `dm_cmrn` | 0.6622 ± 0.0151 | **0.6316 ± 0.0153** | **0.6206 ± 0.0095** | **0.6954 ± 0.0138** | **0.6595 ± 0.0140** | 0.6200 ± 0.0111 | **0.6373 ± 0.0101** |

结论：

- **分类指标两者统计上等价**（clean 0.6317 vs 0.6316；triple_30 0.6205 vs 0.6200），
  差异远小于种子间标准差（σ ≈ 0.011–0.018）；
- **回归指标 `dm_cmrn` 稳定更好**：MAE −0.020、Pearson +0.011、CCC +0.011，
  且**方差显著更小**（CCC σ 0.014 vs 0.041，MAE σ 0.0095 vs 0.0126）→ 训练更稳、强度回归更可靠；
- `dm_cmrn` 平均最优 epoch 5.6 vs `cross_attn` 3.4 → DM-CMRN 需要更多轮才收敛。

⚠️ **单种子结论不可信**：单种子时曾得到 `cross_attn` 0.6613 vs `dm_cmrn` 0.6049（差 0.056），
5 种子下该差距变为 0.0001 —— 论文中的任何“改进”必须给多种子均值±标准差。

### (2) 超参筛选：3 个种子，`--min-delta 0 --patience 6 --epochs 12`（让模型训满）

| 配置 | clean Macro-F1 | triple_30 Macro-F1 | 平均最优 epoch |
|---|---|---|---|
| `rec0.3_span3` | **0.6265 ± 0.0167** | **0.6172 ± 0.0033** | 5.3 |
| `rec0.1_span6` | 0.6259 ± 0.0105 | 0.6168 ± 0.0060 | 4.7 |
| `rec0.05_span6` | 0.6257 ± 0.0110 | 0.6165 ± 0.0113 | 5.3 |
| `rec0.3_span6_long`（放宽早停的对照组） | 0.6254 ± 0.0180 | 0.6132 ± 0.0085 | 5.7 |
| `rec0.3_span12` | 0.6214 ± 0.0112 | 0.6148 ± 0.0063 | 4.7 |

结论：**本轮调参没有带来可测的提升**（全部配置落在 0.621–0.627 / 0.613–0.617 区间内，
差异均小于种子标准差）；把重建权重从 0.3 降到 0.05/0.1、把窗口从 6 改成 3/12、
以及放宽早停让它训满，都无法把分类指标推高。

### (3) 由此得到的判断与下一步

1. **LCR / QAGF 在分类上既未见效也未见害**（与基线等价），在回归强度上有小幅稳定收益；
   “未调好”与“确实无效”之间，目前证据偏向**“在当前架构/数据量下难以兑现”**；
2. 若要继续争取收益，应改结构性因素而不是继续扫这两个超参：
   - 用 **未对齐重采样数据**（`--data-kind resampled`）做同样对比（数据版本差异可能大于模块差异）；
   - 给 LCR 加**更强的监督**（教师特征目标 + 稀疏化/门控约束），或改为“只在被判定为缺失的位置替换”的硬路由；
   - 把 `cross_attn` 的 `use_status_embed` 设为 False，明确双掩码的归属；
3. 报告规范：所有对比都要 5 种子均值±标准差（本目录 `tools/study_runner.py` 已可一键复算）。

## 5.1 本机复现结果（单种子 42，**已被 5.0 取代**，保留用于说明单种子噪声有多大）

`python -u tools\compare_models.py --root outputs --split test` 生成，完整表见
`outputs\comparison_test.md`。场景：`clean` = 无缺失；`triple_30` = 三模态各在有效段内随机缺失 30%。

| 模型 | clean Macro-F1 | triple_30 Macro-F1 | clean MAE |
|---|---|---|---|
| `cross_attn`（Baseline 2，跨模态注意力） | **0.6613** | **0.6480** | 0.6580 |
| `dm_cmrn_global_rec`（全局重建） | 0.6419 | 0.6353 | 0.6347 |
| `dm_cmrn_no_qagf`（去掉质量门控） | 0.6296 | 0.6336 | 0.6491 |
| `dm_cmrn_no_lcr`（去掉局部重建） | 0.6207 | 0.6225 | 0.6409 |
| `teacher`（cross_attn，仅干净输入训练） | 0.6182 | 0.6289 | 0.6254 |
| `dm_cmrn_warmstart`（教师编码器热启动） | 0.6084 | 0.6143 | 0.6137 |
| `early`（Baseline 1） | 0.6106 | 0.6260 | 0.6432 |
| `dm_cmrn`（完整模型） | 0.6049 | 0.6055 | 0.6362 |
| `misa`（Baseline 3） | 0.5987 | 0.5966 | 0.6400 |
| `dm_cmrn_no_dual_mask`（只用 padding mask） | 0.5913 | 0.5721 | 0.6769 |

**如实解读（单种子，尚未做多种子复核，不应作为最终结论）**：

1. **双掩码得到支持**：`no_dual_mask` 是全部配置中最差的（clean 0.5913 / triple_30 0.5721），
   说明“区分 padding 与 missing”确实有用。
2. **LCR 与 QAGF 在当前超参下未兑现收益**：去掉它们反而略好（0.6207 / 0.6296 > 0.6049），
   且“全局重建”优于“局部重建”（0.6419 > 0.6049），与设计预期相反。
3. **Baseline 2 最强**，且它同样带有双掩码状态嵌入（`ModalEncoder` 的 `use_status_embed` 默认为真），
   因此它其实是“共享编码器 + 简单融合”的强基线。
4. 所有 DM-CMRN 变体的 MAE（0.61–0.68）总体优于 `cross_attn`（0.6580），回归指标更稳。

**下一步建议（按优先级）**：

- **多种子复核**：单种子差异 ±0.01 量级（同类实验曾测得种子间标准差约 0.008），
  至少 3–5 个种子再判断 LCR/QAGF 是否真的有增益；
- **调 LCR/QAGF 超参**：`--rec-weight`（0.05/0.1/0.3）、`--ctx-span`（3/6/12）、
  `--distill-weight`、门控宽度；并放宽早停（`--min-delta 0` + `--patience 6`）让模型训满；
- **严格化 Baseline**：若要把“双掩码”作为创新点对比，应让 Baseline 2 使用
  `use_status_embed=False` 的纯单掩码编码器，否则该创新已被基线共享；
- 把 `--warm-start-encoders` 与更长训练结合（本次热启动 0.6084 未超过冷启动 0.6049 的差距在噪声内）。

## 6 输出说明

```
outputs/<run>/best.pt            权重 + 模型参数 + 标准化统计 + 验证指标
outputs/<run>/history.csv        每轮 train / valid_clean / valid_missing 指标与分项损失
outputs/<run>/config.json        训练参数、模型参数、损失权重、最优 epoch
outputs/audit/audit_*.json       数据审计（padding / observed / interior-missing 比例）
outputs/missing_analysis/        missing_grid.csv、missing_analysis.md、curve_ratio.png
outputs/attachment3/             attachment3_predictions.csv、attachment3_summary.json
```

附件3 无真实标签，只报告预测极性/强度、门控权重与缺失区间统计。

多模型对比（把所有 `*/best.pt` 放到同一套场景下评估，直接产出论文表格）：

```powershell
python -u tools\compare_models.py --root outputs --split test
# 产出 outputs\comparison_test.csv 与 comparison_test.md（含 clean 与 triple_30 两列重点场景）
```

## 7 与 `q2_baseline` 的关系与差异

- **数据**：共用附件2 `aligned_50.pkl`；本实现把 `exist / observed / missing` 三类掩码显式化
  （`q2_baseline` 只有单一 mask），并支持长度感知重采样数据版本。
- **结构**：`q2_baseline` 为 Conv1d 残差编码 + 拼接融合；本实现加入**双掩码状态嵌入**、
  **局部连续重建**、**质量门控融合**，同时保留 Baseline 群做对照。
- **训练**：`q2_baseline` 用“连续片段丢弃”隐式模拟缺失；本实现把缺失显式建模为 `missing` 掩码，
  重建损失只作用于缺失位置，并支持教师蒸馏。
- **评估**：本实现自带缺失规律网格（类型/位置/时长/缺失率）与附件3 缺失自动检测，直接对应
  赛题要求；“同一测试划分（727 条）”上的 Accuracy/Macro-F1/MAE/Pearson 与 `q2_baseline` 同口径，
  并额外给出 CCC。

## 8 注意事项

1. 每个 epoch 重新采样缺失模式，同一配置不同种子会有差异；论文中的“改进”应报告多种子均值±标准差。
2. 有效段仅 1 个窗口的极端样本会自动跳过缺失模拟，避免“整模态零观测”导致注意力全屏蔽。
3. 附件3 的 `distilbert` 文本模式与 `q2_baseline/evaluate_attachment3.py` 使用同一模型与修订号，
   便于结果对照；无网络时可用 `--text-mode zero` 仅做缺失文本对照。
4. **`observed` 必须在标准化之前判定**：标准化会把原本的全零行变成 `-mean/std` 的非零值，若放在
   之后再判断，“缺失位置”会被误判为“有观测”。`data.prepare_split` 与 `infer_attachment3.prepare_sample`
   均已在标准化前用原始特征计算 `observed`，并在之后统一置零，保证“特征非零 ⟺ observed”。
5. **附件3 的 `--exist-mode`（重要）**：附件3 的 pkl 只有 `text_bert`/`raw_text` + `audio`/`vision`
   （带前置批次维），**没有长度或掩码字段**，且其非零内容只覆盖时间轴的前一段（例如附件3_01 的
   音频非零窗为 1–6，其余全零）。因此“长度边界外的零”有两种解读，必须并行列报告：
   - `--exist-mode span`：视为长度填充（与附件2 训练口径一致）→ 缺失计数为 0；
   - `--exist-mode full`（默认）：视为“本该存在但缺失”→ 例如 aligned 样本音频平均缺失 33.2 窗，
     模型会走局部重建补全流程。
   哪种解读更接近出题人意图需要结合附件3 的生成说明确认；本实现把两种口径都保留并可复算。
   text 模态在 `distilbert` 模式下由 token 数决定有效位置，故其缺失计数为 0（`zero` 模式则整体缺失）。


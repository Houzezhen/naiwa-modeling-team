"""Generate auditable, publication-ready figures from the frozen validation checkpoint.

Figure contract: validation superiority over a fixed baseline is descriptive, not
an external-generalization or significance claim. All quantitative plots use real
validation inference or recorded validation JSON; no synthetic observations.
Backend: Python/matplotlib. Output: standalone SVG, PDF and 300 dpi PNG.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch
from torch.nn import functional as functional
from nature_plot_style import (BLUE, TEAL, ORANGE, GOLD, PURPLE, GRAY, INK, PALE,
                               CHINESE_CLASSES, CHINESE_MODALITIES, apply_style)


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
OUT = PAPER / 'figures'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'q3_explainability'))
import q3_pipeline as pipeline

CLASSES = pipeline.CLASSES
MODALITIES = pipeline.MODALITIES
apply_style()


def save(fig, filename):
    fig.savefig(OUT / f'{filename}.svg', bbox_inches='tight')
    fig.savefig(OUT / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def write_rows(filename, rows, columns):
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def model_comparison():
    final = json.loads((ROOT / 'q3_explainability' / 'q3_aligned_text_bert_final' / 'validation.json').read_text(encoding='utf-8'))
    baseline_report = (ROOT / 'q2_baseline' / 'REPORT.md').read_text(encoding='utf-8')
    baseline_line = next(line for line in baseline_report.splitlines() if line.startswith('| Fixed aligned baseline |'))
    cells = [cell.strip().strip('`*') for cell in baseline_line.split('|')]
    baseline = dict(zip(['accuracy', 'f1_macro', 'mae', 'pearson'], map(float, cells[3:7])))
    keys = ['accuracy', 'f1_macro', 'mae', 'pearson']
    names = ['准确率', '宏平均调和分数', '平均绝对误差', '皮尔逊相关']
    rows = [{'metric': name, 'baseline': baseline[key], 'final': final[key],
             'delta_final_minus_baseline': final[key] - baseline[key],
             'baseline_input': 'aligned_features.pkl', 'final_input': 'aligned_text_bert_features.pkl',
             'baseline_source': 'q2_baseline/REPORT.md', 'split': 'attachment2 validation (n=728)'}
            for key, name in zip(keys, names)]
    write_rows('source_model_metrics.csv', rows, list(rows[0]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.35), sharey=False)
    accents = [BLUE, TEAL, PURPLE, ORANGE]
    for axis, row, accent in zip(axes, rows, accents):
        values = [row['baseline'], row['final']]
        axis.bar([0, 1], values, color=[GRAY, TEAL], width=.58,
                 edgecolor=INK, linewidth=.45)
        axis.set_xticks([0, 1]); axis.set_xticklabels(['固定基线', '最终模型'])
        value_range = max(values) - min(values)
        local_pad = max(value_range * .80, .010)
        axis.set_ylim(min(values) - local_pad, max(values) + local_pad)
        axis.set_title(row['metric'], color=accent)
        axis.grid(axis='y', alpha=.55)
        for position, value in enumerate(values):
            axis.text(position, value + local_pad * .13, f'{value:.3f}',
                      ha='center', fontsize=8, weight='semibold')
        gain = -row['delta_final_minus_baseline'] if row['metric'] == '平均绝对误差' else row['delta_final_minus_baseline']
        axis.text(.5, .96, f'改善 {gain:+.4f}', transform=axis.transAxes,
                  ha='center', va='top', color=accent, fontsize=8, weight='bold')
        axis.text(.98, .04, '局部纵轴', transform=axis.transAxes,
                  ha='right', va='bottom', color=GRAY, fontsize=6.5)
    axes[0].set_ylabel('验证集指标值')
    fig.suptitle('同一验证划分（728例）；文本特征版本不同；局部纵轴放大近似值差异', fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, 'fig_main_performance')


def missing_heatmap():
    report = json.loads((ROOT / 'q2_baseline' / 'aligned_text_bert_run_bidir' / 'fusion_validation.json').read_text(encoding='utf-8'))
    clean = report['results']['clean']['f1_macro']
    rows = []
    for modality in MODALITIES:
        for position in ['start', 'middle', 'end']:
            for window_count in [5, 15, 25]:
                key = f'{modality}_{position}_{window_count}'
                rows.append({'modality': modality, 'position': position, 'masked_windows': window_count,
                             'clean_f1': clean, 'masked_f1': report['results'][key]['f1_macro'],
                             'delta_f1': report['results'][key]['f1_macro'] - clean})
    write_rows('source_missing_scenarios.csv', rows, list(rows[0]))
    bound = max(abs(row['delta_f1']) for row in rows)
    norm = SymLogNorm(linthresh=.01, linscale=1, vmin=-bound, vmax=bound)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for axis, modality, accent in zip(axes, MODALITIES, [ORANGE, TEAL, BLUE]):
        grid = np.array([[next(row['delta_f1'] for row in rows if row['modality'] == modality
                          and row['position'] == position and row['masked_windows'] == count)
                          for count in [5, 15, 25]] for position in ['start', 'middle', 'end']])
        image = axis.imshow(grid, cmap='RdBu', norm=norm, aspect='auto')
        axis.set_xticks(range(3)); axis.set_xticklabels(['5', '15', '25'])
        axis.set_yticks(range(3)); axis.set_yticklabels(['起始', '中段', '末段'])
        axis.set_xlabel('被遮挡窗口数（共50个）')
        axis.set_title(CHINESE_MODALITIES[modality], color=accent)
        for position in np.ndindex(grid.shape):
            value = grid[position]
            axis.text(position[1], position[0], f'{value:+.3f}', color='white' if abs(value) > .6 * bound else INK,
                      ha='center', va='center', fontsize=8)
    axes[0].set_ylabel('遮挡位置')
    fig.colorbar(image, ax=axes, fraction=.023, pad=.025,
                 ticks=[-.15, -.05, -.01, 0, .01, .05, .15],
                 label='相对完整输入的分数变化（对称对数色阶）')
    fig.suptitle('统一对齐候选模型的缺失敏感性（验证集728例）', fontsize=11, y=.99)
    fig.subplots_adjust(left=.08, right=.88, bottom=.19, top=.78, wspace=.24)
    save(fig, 'fig_missing_heatmap')


def infer_validation():
    pipeline.seed_everything(42)
    model, checkpoint = pipeline.load_model(ROOT / 'q3_explainability' / 'q3_aligned_text_bert_final' / 'q3_explainable_best.pt', 'cpu')
    data = pipeline.load_pickle(ROOT / 'emotion-transfer-pilot' / 'aligned_text_bert_features.pkl')
    features, masks, targets, ids = pipeline.convert(data['valid'], checkpoint['normalizers'])
    dataset = (features, masks, targets, ids)
    predictions, strengths, attention, gates, probabilities = [], [], [], [], []
    with torch.no_grad():
        for batch_features, batch_masks, _ in pipeline.make_batches(dataset, 64, 'cpu', False):
            logits, regression, gate, temporal = model(batch_features, batch_masks)
            probs = functional.softmax(logits, dim=-1)
            predictions.extend(probs.argmax(-1).cpu().tolist())
            strengths.extend(regression.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            gates.extend(gate.cpu().tolist())
            attention.extend(torch.stack(temporal, dim=1).cpu().tolist())
    truth = targets['class'].numpy()
    actual = targets['regression'].numpy()
    predicted = np.array(predictions)
    estimate = np.array(strengths)
    metrics = pipeline.score_metrics(truth, predicted, actual, estimate)
    for key, recorded in checkpoint['validation'].items():
        if abs(metrics[key] - recorded) > 1e-5:
            raise AssertionError(f'Reproduced {key} differs from checkpoint: {metrics[key]} vs {recorded}')
    rows = [{'sample_id': str(ids[index]), 'true_class': CLASSES[int(truth[index])],
             'predicted_class': CLASSES[int(predicted[index])], 'true_intensity': float(actual[index]),
             'predicted_intensity': float(estimate[index])} for index in range(len(truth))]
    write_rows('source_validation_predictions.csv', rows, list(rows[0]))
    return model, dataset, truth, actual, predicted, estimate, np.array(attention), np.array(gates), np.array(probabilities), metrics


def confusion_matrix(truth, predicted):
    matrix = np.zeros((3, 3), dtype=int)
    for actual_label, predicted_label in zip(truth, predicted):
        matrix[actual_label, predicted_label] += 1
    write_rows('source_confusion_matrix.csv', [dict(true_class=CLASSES[index], **{name: int(matrix[index, col]) for col, name in enumerate(CLASSES)}) for index in range(3)], ['true_class', *CLASSES])
    fig, axis = plt.subplots(figsize=(5.1, 4.15))
    axis.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
    for position in np.ndindex(matrix.shape):
        count = matrix[position]
        axis.text(position[1], position[0], str(count), ha='center', va='center', fontsize=13,
                  color='white' if count > matrix.max() * .56 else INK)
    axis.set_xticks(range(3)); axis.set_xticklabels([CHINESE_CLASSES[name] for name in CLASSES])
    axis.set_yticks(range(3)); axis.set_yticklabels([CHINESE_CLASSES[name] for name in CLASSES])
    axis.set_xlabel('预测类别'); axis.set_ylabel('真实类别')
    axis.set_title('最终模型验证集混淆矩阵（728例）')
    fig.tight_layout()
    save(fig, 'fig_validation_confusion')


def regression_scatter(actual, estimate, metrics):
    fig, axis = plt.subplots(figsize=(5.6, 4.25))
    lower = min(actual.min(), estimate.min()) - .1
    upper = max(actual.max(), estimate.max()) + .1
    axis.scatter(actual, estimate, s=13, alpha=.48, color=BLUE, linewidths=0)
    axis.plot([lower, upper], [lower, upper], linestyle='--', color=ORANGE, lw=1.2, label='完全一致参考线')
    axis.set_xlim(lower, upper); axis.set_ylim(lower, upper)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('真实情感强度'); axis.set_ylabel('预测情感强度')
    axis.text(.03, .97, f"平均绝对误差：{metrics['mae']:.3f}\n皮尔逊相关：{metrics['pearson']:.3f}",
              transform=axis.transAxes, va='top', color=INK, fontsize=9,
              bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': '#D5DBE5', 'alpha': .95})
    axis.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    save(fig, 'fig_validation_regression')


def representative(model, dataset, truth, predicted, attention, gates, probabilities):
    features, masks, targets, ids = dataset
    eligible = [index for index in range(len(truth)) if truth[index] == predicted[index]
                and all(bool(masks[modality][index].any()) for modality in MODALITIES)]
    for sample_index in eligible:
        candidate_features = {modality: values[sample_index:sample_index + 1] for modality, values in features.items()}
        candidate_masks = {modality: values[sample_index:sample_index + 1] for modality, values in masks.items()}
        explanation = pipeline.explain_sample(model, candidate_features, candidate_masks, 'cpu', 1)
        if max(row['occlusion_drop'] for row in explanation['evidence']) >= .01:
            break
    else:
        raise AssertionError('No correctly classified validation sample meets the stated illustration criterion')
    sample_display = f'第{sample_index + 1}条'
    sample_features = {modality: values[sample_index:sample_index + 1] for modality, values in features.items()}
    sample_masks = {modality: values[sample_index:sample_index + 1] for modality, values in masks.items()}
    target_class = int(predicted[sample_index])
    selected_attention = attention[sample_index]
    fig, axis = plt.subplots(figsize=(10.4, 2.85))
    image = axis.imshow(selected_attention, cmap='YlGnBu', aspect='auto', interpolation='nearest', vmin=0)
    axis.set_yticks(range(3)); axis.set_yticklabels([CHINESE_MODALITIES[name] for name in MODALITIES])
    axis.set_xticks([0, 9, 19, 29, 39, 49]); axis.set_xticklabels(['1', '10', '20', '30', '40', '50'])
    axis.set_xlabel('统一时间窗序号（从1开始）')
    axis.set_title(f'验证样本{sample_display}：三模态掩码时间注意力')
    for row_index, modality in enumerate(MODALITIES):
        top_index = int(np.argmax(selected_attention[row_index]))
        axis.plot(top_index, row_index, marker='s', markersize=8, markerfacecolor='none',
                  markeredgecolor=ORANGE, markeredgewidth=1.6)
        for missing_index in np.flatnonzero(~masks[modality][sample_index].numpy()):
            axis.plot(missing_index, row_index, '.', color=GRAY, markersize=1)
    fig.colorbar(image, ax=axis, label='模态内注意力权重', fraction=.022, pad=.025)
    fig.tight_layout()
    save(fig, 'fig_temporal_attention')
    stages = [float(probabilities[sample_index, target_class])]
    modality_order = sorted(MODALITIES, key=lambda name: -explanation['contributions'][name])
    changed_features = {name: value.clone() for name, value in sample_features.items()}
    changed_masks = {name: value.clone() for name, value in sample_masks.items()}
    with torch.no_grad():
        for modality in modality_order:
            changed_features[modality].zero_()
            changed_masks[modality].fill_(False)
            logits, _, _, _ = model(changed_features, changed_masks)
            stages.append(float(functional.softmax(logits, -1)[0, target_class]))
    source = [{'stage': 'full', 'target_class_probability': stages[0], 'change': 0.0}]
    source.extend({'stage': f'mask_{name}', 'target_class_probability': stages[index + 1],
                   'change': stages[index + 1] - stages[index]} for index, name in enumerate(modality_order))
    write_rows('source_sequential_ablation.csv', source, list(source[0]))
    fig, axis = plt.subplots(figsize=(8.7, 3.55))
    axis.barh(0, stages[0], height=.56, color=BLUE)
    axis.text(stages[0] + .012, 0, f'{stages[0]:.3f}', va='center', color=INK)
    for index, modality in enumerate(modality_order, start=1):
        before, after = stages[index - 1:index + 1]
        axis.barh(index, abs(after - before), left=min(after, before), height=.56,
                  color=ORANGE if after < before else TEAL)
        axis.plot([before, before], [index - 1 + .28, index - .28], color=GRAY, lw=.8)
        axis.text(max(before, after) + .012, index, f'{after - before:+.3f}', va='center', color=INK)
    axis.set_yticks(range(4)); axis.set_yticklabels(['完整输入', *[f'继续遮挡{CHINESE_MODALITIES[name]}' for name in modality_order]])
    axis.invert_yaxis(); axis.set_xlim(0, 1.05)
    axis.set_xlabel(f'原预测类别（{CHINESE_CLASSES[CLASSES[target_class]]}）的置信度')
    axis.set_title(f'验证样本{sample_display}：顺序模态消融')
    axis.grid(axis='x', alpha=.15)
    fig.tight_layout()
    save(fig, 'fig_ablation_waterfall')
    evidence = explanation['evidence']
    top_evidence = sorted(evidence, key=lambda row: row['evidence_score'], reverse=True)[:3]
    fig, axis = plt.subplots(figsize=(9.5, 3.0))
    axis.set_xlim(0, 10); axis.set_ylim(0, 3.3); axis.axis('off')
    axis.text(.2, 2.94, f'验证集代表性样本：{sample_display}', fontsize=12, weight='bold', color=BLUE)
    axis.text(.2, 2.46, f"真实：{CHINESE_CLASSES[CLASSES[int(truth[sample_index])]]}  "
              f"预测：{CHINESE_CLASSES[CLASSES[target_class]]}  置信度：{stages[0]:.3f}", color=INK)
    axis.text(.2, 1.94, '模态门控权重', weight='bold', color=INK)
    for row_index, modality in enumerate(MODALITIES):
        axis.text(.2, 1.55 - .36 * row_index,
                  f'{CHINESE_MODALITIES[modality]}  {gates[sample_index, row_index]:.3f}', color=INK)
        axis.barh(1.57 - .36 * row_index, gates[sample_index, row_index], left=2.15, height=.16,
                  color=[ORANGE, TEAL, BLUE][row_index])
    axis.text(5.0, 1.94, '局部遮挡敏感证据：前三窗口', weight='bold', color=INK)
    for row_index, entry in enumerate(top_evidence):
        axis.text(5.0, 1.55 - .36 * row_index,
                  f"{row_index + 1}. {CHINESE_MODALITIES[entry['modality']]}  第{entry['window_index'] + 1:02d}窗"
                  f"   得分 {entry['evidence_score']:.5f}   下降 {entry['occlusion_drop']:+.3f}",
                  fontsize=8.5, color=INK)
    axis.text(.2, .17, '展示规则：首条分类正确、三模态均有效且最高注意力窗口遮挡后置信度下降至少0.01的样本。',
              fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, 'fig_explanation_card')
    (OUT / 'source_representative_sample.json').write_text(json.dumps({
        'sample_index': sample_index, 'sample_id': str(ids[sample_index]),
        'selection_rule': 'First correctly classified validation sample with all modalities present and top-window occlusion drop >= 0.01; illustration only',
        'true_class': CLASSES[int(truth[sample_index])], 'predicted_class': CLASSES[target_class],
        'class_probability': stages[0], 'gate': explanation['contributions'], 'evidence': evidence,
    }, indent=2), encoding='utf-8')


def main():
    model_comparison()
    missing_heatmap()
    model, dataset, truth, actual, predicted, estimate, attention, gates, probabilities, metrics = infer_validation()
    confusion_matrix(truth, predicted)
    regression_scatter(actual, estimate, metrics)
    representative(model, dataset, truth, predicted, attention, gates, probabilities)
    print(json.dumps({'validation_n': len(truth), 'metrics_reproduced': metrics,
                      'figures': sorted(path.name for path in OUT.glob('fig_*.svg'))}, indent=2))


if __name__ == '__main__':
    main()

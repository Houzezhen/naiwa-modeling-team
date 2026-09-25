from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import image as mpimg
from nature_plot_style import (BLUE, TEAL, ORANGE, GOLD, PURPLE, GRAY,
                               INK as DARK, CHINESE_CLASSES, apply_style)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
apply_style()
plt.rcParams.update({'legend.fontsize': 8, 'figure.dpi': 160, 'savefig.bbox': 'tight'})

# 1. End-to-end workflow diagram.
fig, ax = plt.subplots(figsize=(13, 5.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis('off')
boxes = [
    (0.3, 3.35, 2.0, 1.15, 'Raw data\ntext / audio / video', '#E8EEF8'),
    (2.75, 3.35, 2.0, 1.15, 'Feature extraction\n768 / 74 / 35', '#E8F5F2'),
    (5.2, 3.35, 2.0, 1.15, '50-window\ncommon timeline', '#FFF3DF'),
    (7.65, 3.35, 2.0, 1.15, 'Masked\nnormalization', '#F4EAF8'),
    (10.1, 3.35, 2.0, 1.15, 'Prediction\n+ evidence', '#FDECEC'),
]
for x,y,w,h,t,c in boxes:
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.04,rounding_size=0.08', fc=c, ec=DARK, lw=1.0))
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', color=DARK, weight='bold')
for i in range(len(boxes)-1):
    x1=boxes[i][0]+boxes[i][2]+0.05; x2=boxes[i+1][0]-0.05; y=3.92
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y), arrowstyle='-|>', mutation_scale=14, lw=1.4, color=GRAY))
ax.text(0.4, 2.55, 'Quality audit', weight='bold', color=BLUE)
ax.text(0.4, 2.15, 'coverage · monotonicity · non-overlap · mask consistency', color=DARK)
ax.text(4.15, 2.55, 'Predictive model', weight='bold', color=TEAL)
ax.text(4.15, 2.15, 'three BiGRU encoders → temporal attention → modality gate → fusion', color=DARK)
ax.text(8.45, 2.55, 'Interpretability', weight='bold', color=ORANGE)
ax.text(8.45, 2.15, 'modality ablation · window occlusion · keyframe traceability', color=DARK)
ax.text(6.5, 0.75, 'Unified auditable pipeline for complex-scene multimodal emotion recognition', ha='center', fontsize=12, weight='bold', color=BLUE)
fig.savefig(OUT/'fig_pipeline_legacy.pdf'); fig.savefig(OUT/'fig_pipeline_legacy.png', dpi=300); plt.close(fig)

# 2. Alignment coverage and status.
fig, axes = plt.subplots(1,2,figsize=(11,4.6), gridspec_kw={'width_ratios':[1.25,1]})
mods=['文本','语音','视觉','语音候选']; cov=[58.68,99.00,99.80,86.74]
colors=[ORANGE,TEAL,BLUE,GOLD]
axes[0].bar(mods,cov,color=colors,width=.62)
axes[0].set_ylim(0,105); axes[0].set_ylabel('覆盖率（%）'); axes[0].set_title('公共窗口模态覆盖率')
for i,v in enumerate(cov): axes[0].text(i,v+2,f'{v:.2f}',ha='center',fontsize=8)
axes[0].grid(axis='y',alpha=.2)
status=['候选对齐（未人工确认）','低置信度','语音与转写冲突']
vals=[80,18,2]
axes[1].barh(status[::-1], vals[::-1], color=[ORANGE,GOLD,BLUE], height=.52)
axes[1].set_xlim(0,100); axes[1].set_xlabel('样本数（共100条）')
axes[1].set_title('文本对齐质量抽样审计')
axes[1].grid(axis='x', alpha=.55)
for i, value in enumerate(vals[::-1]): axes[1].text(value + 1.4, i, str(value), va='center', fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'fig_alignment_quality.pdf'); fig.savefig(OUT/'fig_alignment_quality.png', dpi=300); plt.close(fig)

# 2b. Class composition in the supervised split.
class_counts = {
    '训练集': {'Negative': 967, 'Neutral': 758, 'Positive': 1670},
    '验证集': {'Negative': 206, 'Neutral': 184, 'Positive': 338},
    '测试集': {'Negative': 207, 'Neutral': 158, 'Positive': 362},
}
class_colors = {'Negative': ORANGE, 'Neutral': GRAY, 'Positive': TEAL}
fig, ax = plt.subplots(figsize=(9.2, 4.4))
bottom = np.zeros(3)
splits = list(class_counts)
for cls in ['Negative', 'Neutral', 'Positive']:
    values = np.array([class_counts[s][cls] for s in splits], dtype=float)
    totals = np.array([sum(class_counts[s].values()) for s in splits], dtype=float)
    shares = 100 * values / totals
    ax.bar(splits, shares, bottom=bottom, color=class_colors[cls], width=0.58, label=CHINESE_CLASSES[cls])
    for i, (b, share) in enumerate(zip(bottom, shares)):
        if share >= 8:
            ax.text(i, b + share / 2, f'{share:.1f}%', ha='center', va='center', color='white', fontsize=8, weight='bold')
    bottom += shares
ax.set_ylim(0, 100); ax.set_ylabel('类别占比（%）'); ax.set_title('附件2监督数据的类别构成', pad=32)
ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(); fig.savefig(OUT/'fig_class_distribution.pdf'); fig.savefig(OUT/'fig_class_distribution.png', dpi=300); plt.close(fig)

# 3. Typical timeline and frame snapshots.
frames_dir=ROOT/'E题复杂场景下多模态情感识别的数学建模与算法设计'/'question1_alignment'/'output_speech_aware'/'typical_sample_frames'
frame_names=['bin_00_time_0.067s.jpg','bin_12_time_1.667s.jpg','bin_25_time_3.400s.jpg','bin_37_time_5.000s.jpg','bin_49_time_6.567s.jpg']
frame_times=[0.067,1.667,3.400,5.000,6.567]
fig=plt.figure(figsize=(11,4.1))
grid=fig.add_gridspec(2,5,height_ratios=[.55,1.15])
timeline_ax=fig.add_subplot(grid[0,:])
frame_axes=[fig.add_subplot(grid[1,i]) for i in range(5)]
T=6.644987; W=T/50
for k in range(50): timeline_ax.barh(.5, W*.94, left=k*W, height=.27, color='#E4EFF1', edgecolor='white', linewidth=.25)
for t in frame_times: timeline_ax.axvline(t,color=ORANGE,lw=1.2,ls='--')
timeline_ax.set_xlim(0,T); timeline_ax.set_ylim(0,1); timeline_ax.set_yticks([]); timeline_ax.set_xlabel('视频时间（秒）'); timeline_ax.set_title('典型样本的50个等宽公共时间窗（时长6.645秒）')
timeline_ax.text(.01,.82,'色块表示公共窗口；虚线对应下方视频帧',transform=timeline_ax.transAxes,color=GRAY,fontsize=8)
for ax,t,n in zip(frame_axes, frame_times, frame_names):
    im=mpimg.imread(frames_dir/n); ax.imshow(im[:int(im.shape[0]*.87)]); ax.set_title(f'{t:.3f}秒',fontsize=8); ax.axis('off')
fig.tight_layout()
fig.savefig(OUT/'fig_typical_timeline.pdf'); fig.savefig(OUT/'fig_typical_timeline.png', dpi=300); plt.close(fig)

# 4. Model architecture.
fig, ax = plt.subplots(figsize=(13,6.3)); ax.set_xlim(0,13); ax.set_ylim(0,6.3); ax.axis('off')
def box(x,y,w,h,text,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.06',fc=fc,ec=DARK,lw=.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5,color=DARK,weight='bold')
for x,label,c in [(0.3,'Text\n50×768','#E8EEF8'),(0.3,'Audio\n50×74','#E8F5F2'),(0.3,'Vision\n50×35','#FFF3DF')]: pass
ys=[4.55,3.25,1.95]; labels=['Text\n50×768','Audio\n50×74','Vision\n50×35']; cs=['#E8EEF8','#E8F5F2','#FFF3DF']
for y,l,c in zip(ys,labels,cs): box(.35,y,1.3,.72,l,c); box(2.0,y,1.65,.72,'Projection\n+ mask', '#F3F4F6'); box(4.05,y,1.7,.72,'2-layer\nBiGRU', '#E8EEF8'); box(6.2,y,1.75,.72,'Temporal\nattention', '#F4EAF8')
    
for y in ys:
    for x1,x2 in [(1.65,2.0),(3.65,4.05),(5.75,6.2)]: ax.add_patch(FancyArrowPatch((x1,y+.36),(x2,y+.36),arrowstyle='-|>',mutation_scale=11,lw=1,color=GRAY))
box(8.35,3.15,1.8,1.7,'Modality gate\nβtext, βaudio,\nβvision','#FDECEC')
for y in ys: ax.add_patch(FancyArrowPatch((7.95,y+.36),(8.35,4.0),connectionstyle='arc3,rad=.15',arrowstyle='-|>',mutation_scale=10,lw=1,color=GRAY))
box(10.55,3.15,1.85,1.7,'Fusion MLP\nLayerNorm + GELU\n+ Dropout','#E8EEF8')
ax.add_patch(FancyArrowPatch((10.15,4.0),(10.55,4.0),arrowstyle='-|>',mutation_scale=12,lw=1.2,color=GRAY))
box(11.0,1.05,1.55,.78,'Class head\n3 classes','#E8F5F2'); box(9.05,1.05,1.55,.78,'Regression head\nstrength','#FFF3DF')
ax.add_patch(FancyArrowPatch((11.45,3.15),(11.75,1.83),connectionstyle='arc3,rad=.15',arrowstyle='-|>',mutation_scale=10,lw=1,color=GRAY)); ax.add_patch(FancyArrowPatch((11.05,3.15),(9.8,1.83),connectionstyle='arc3,rad=-.15',arrowstyle='-|>',mutation_scale=10,lw=1,color=GRAY))
ax.text(6.5,.3,'Hierarchical temporal–modal fusion with two supervised outputs',ha='center',fontsize=12,weight='bold',color=BLUE)
fig.savefig(OUT/'fig_model_architecture_legacy.pdf'); fig.savefig(OUT/'fig_model_architecture_legacy.png',dpi=300); plt.close(fig)

# 5. Candidate comparison on validation.
models=['固定基线','重采样基线','双向循环候选','旧残差融合\n（不合规）','最终注意力门控']
metrics={
'准确率':[.612637,.614011,.618132,.622253,.637363],
'宏平均调和分数':[.588115,.595798,.598285,.610330,.601580],
'平均绝对误差':[.616443,.614132,.621818,.623941,.612262],
'皮尔逊相关':[.624582,.620838,.588408,.601396,.629627]}
fig, axes=plt.subplots(2,2,figsize=(11,7))
for ax,(name,vals) in zip(axes.flat,metrics.items()):
    colors=[GRAY,PURPLE,BLUE,ORANGE,TEAL]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor=DARK, linewidth=.45)
    ax.axhline(vals[0], color=GRAY, ls='--', lw=.8)
    local_pad=max((max(vals)-min(vals))*.65, .012)
    ax.set_title(name); ax.set_xticks(range(len(vals))); ax.set_xticklabels(models,fontsize=8)
    ax.set_ylim(min(vals)-local_pad, max(vals)+local_pad); ax.grid(axis='y',alpha=.55)
    ax.text(.98, .04, '局部纵轴', transform=ax.transAxes, ha='right', va='bottom', color=GRAY, fontsize=6.5)
    for i,v in enumerate(vals): ax.text(i,v+local_pad*.12,f'{v:.3f}',ha='center',fontsize=7,weight='semibold')
fig.suptitle('验证集候选模型比较；局部纵轴放大近似值差异；旧残差方案不满足统一输入要求',y=1.01,fontsize=11,weight='bold')
fig.tight_layout(); fig.savefig(OUT/'fig_model_comparison.pdf'); fig.savefig(OUT/'fig_model_comparison.png',dpi=300); plt.close(fig)

# 5b. Directional metric gains: validation selection versus independent test review.
metric_labels = ['准确率', '宏平均调和分数', '平均绝对误差\n（反向计优）', '皮尔逊相关']
valid_base = np.array([0.612637, 0.588115, 0.616443, 0.624582])
valid_final = np.array([0.637363, 0.601580, 0.612262, 0.629627])
test_base = np.array([0.650619, 0.604127, 0.657608, 0.653185])
test_final = np.array([0.664374, 0.590653, 0.658821, 0.653034])
def directional_gain(base, final):
    return np.array([final[0] - base[0], final[1] - base[1], base[2] - final[2], final[3] - base[3]]) * 100
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
for ax, title, gain in zip(axes, ['验证集：模型选择', '测试集：独立复核'], [directional_gain(valid_base, valid_final), directional_gain(test_base, test_final)]):
    colors = [TEAL if v >= 0 else ORANGE for v in gain]
    bars = ax.bar(metric_labels, gain, color=colors, width=.60)
    ax.axhline(0, color=DARK, lw=.8); ax.grid(axis='y', alpha=.55)
    ax.set_title(title, fontsize=10); ax.set_ylabel('同一指标的方向性变化（百分点）', fontsize=9)
    ax.set_ylim(-1.70, 2.85)
    ax.tick_params(axis='x', labelrotation=12, labelsize=10)
    for bar, value in zip(bars, gain):
        y = value + (0.10 if value >= 0 else -0.10)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f'{value:+.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
fig.suptitle('最终模型相对固定基线的指标变化', y=1.02, fontsize=11, weight='bold')
fig.tight_layout(); fig.savefig(OUT/'fig_metric_delta.pdf'); fig.savefig(OUT/'fig_metric_delta.png', dpi=300); plt.close(fig)

# 6. Missing-modality sensitivity from the strict aligned candidate.
run_path=ROOT/'q2_baseline'/'aligned_text_bert_run_bidir'/'fusion_validation.json'
obj=json.loads(run_path.read_text(encoding='utf-8'))
clean=obj['results']['clean']
scenarios=['text_middle_15','audio_middle_15','vision_middle_15']
labels=['遮挡文本','遮挡语音','遮挡视觉']
fig, axes=plt.subplots(1,2,figsize=(10.5,4.4))
for metric,ax in [('accuracy',axes[0]),('f1_macro',axes[1])]:
    base=clean[metric]; drops=[base-obj['results'][s][metric] for s in scenarios]
    ax.bar(labels,drops,color=[ORANGE,TEAL,BLUE]); ax.set_ylabel('相对完整输入的下降'); ax.set_title(f'{"准确率" if metric == "accuracy" else "宏平均调和分数"}：中段遮挡15窗'); ax.grid(axis='y',alpha=.55)
    for i,v in enumerate(drops): ax.text(i,v+.001,f'{v:.3f}',ha='center',fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'fig_missing_modality.pdf'); fig.savefig(OUT/'fig_missing_modality.png',dpi=300); plt.close(fig)

# 6b. Quantitative parameter audit for the complexity discussion.
module_labels = ['特征投影', '双向时序编码', '编码器归一化', '时间注意力', '门控与融合', '双任务输出头']
module_values = np.array([112640, 446976, 768, 1155, 182147, 516])
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={'width_ratios': [1.0, 1.35]})
axes[0].bar(['固定基线', '最终模型'], [106372, 744202], color=[GRAY, TEAL], width=.55)
axes[0].set_ylabel('可训练参数量'); axes[0].set_title('总参数量对比'); axes[0].grid(axis='y', alpha=.55)
for i, value in enumerate([106372, 744202]): axes[0].text(i, value + 18000, f'{value:,}', ha='center', fontsize=8)
axes[1].barh(module_labels, module_values, color=[ORANGE, BLUE, GRAY, GOLD, PURPLE, TEAL])
axes[1].set_xlabel('可训练参数量'); axes[1].set_title('最终模型参数构成'); axes[1].grid(axis='x', alpha=.55)
for i, value in enumerate(module_values): axes[1].text(value + 10000, i, f'{value/1000:.1f}千', va='center', fontsize=8)
fig.suptitle('实现模型的参数量审计', y=1.02, fontsize=11, weight='bold')
fig.tight_layout(); fig.savefig(OUT/'fig_complexity_params.pdf'); fig.savefig(OUT/'fig_complexity_params.png', dpi=300); plt.close(fig)

# 7. Explainability summary and modality contributions.
q3exp=json.loads((ROOT/'q3_explainability'/'q3_aligned_text_bert_final'/'explanation_validation.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((ROOT/'q3_explainability'/'q3_aligned_text_bert_final'/'attachment4'/'q3_predictions.csv').open(encoding='utf-8-sig')))
contrib=[np.mean([float(r[f'{m}_contribution']) for r in rows]) for m in ['text','audio','vision']]
fig, axes=plt.subplots(1,2,figsize=(10.5,4.4))
axes[0].bar(['最高注意力窗口','随机窗口'],[q3exp['mean_top_attention_drop'],q3exp['mean_random_drop']],color=[ORANGE,GRAY])
axes[0].set_ylabel('平均置信度下降'); axes[0].set_title('窗口遮挡的解释有效性（128例）'); axes[0].grid(axis='y',alpha=.55)
for i,v in enumerate([q3exp['mean_top_attention_drop'],q3exp['mean_random_drop']]): axes[0].text(i,v+.0002,f'{v:.4f}',ha='center')
axes[1].bar(['文本','语音','视觉'],contrib,color=[ORANGE,TEAL,BLUE]); axes[1].set_ylim(0,1); axes[1].set_ylabel('平均门控权重'); axes[1].set_title('附件4的模态贡献（20例）'); axes[1].grid(axis='y',alpha=.55)
for i,v in enumerate(contrib): axes[1].text(i,v+.02,f'{v:.3f}',ha='center')
fig.tight_layout(); fig.savefig(OUT/'fig_explainability.pdf'); fig.savefig(OUT/'fig_explainability.png',dpi=300); plt.close(fig)

# 8. Attachment 4 predictions.
classes=['Negative','Neutral','Positive']; counts={c:sum(r['predicted_class']==c for r in rows) for c in classes}
strength=[float(r['predicted_strength']) for r in rows]
fig, axes=plt.subplots(1,2,figsize=(10.5,4.4))
axes[0].bar([CHINESE_CLASSES[name] for name in classes],[counts[c] for c in classes],color=[ORANGE,GRAY,TEAL]); axes[0].set_ylabel('样本数'); axes[0].set_title('附件4预测情感极性（20例）'); axes[0].grid(axis='y',alpha=.55)
for i,c in enumerate(classes): axes[0].text(i,counts[c]+.2,str(counts[c]),ha='center')
axes[1].plot(range(1,len(strength)+1),strength,'o-',color=BLUE,ms=4); axes[1].axhline(np.mean(strength),color=ORANGE,ls='--',label=f'平均值：{np.mean(strength):.3f}'); axes[1].set_xlabel('样本序号'); axes[1].set_ylabel('预测情感强度'); axes[1].set_title('附件4连续情感强度'); axes[1].legend(); axes[1].grid(alpha=.55)
fig.tight_layout(); fig.savefig(OUT/'fig_attachment4.pdf'); fig.savefig(OUT/'fig_attachment4.png',dpi=300); plt.close(fig)

# 9. Six representative evidence frames.
key_dir=ROOT/'q3_explainability'/'q3_aligned_text_bert_final'/'attachment4'/'video_keyframes'
keys=sorted(key_dir.glob('*.jpg'))[:6]
fig, axes=plt.subplots(2,3,figsize=(10.5,6.2))
for ax,p in zip(axes.flat,keys):
    image=mpimg.imread(p); ax.imshow(image[:int(image.shape[0]*.87)]); parts=p.stem.split('_')
    ax.set_title(f'样本{parts[0]} · 排名{parts[2]} · 窗口{parts[4]}',fontsize=8); ax.axis('off')
for ax in axes.flat[len(keys):]: ax.axis('off')
fig.suptitle('高证据窗口对应的原始视频关键帧（仅裁切画面下缘）',y=.99,fontsize=11,weight='bold')
fig.tight_layout(); fig.savefig(OUT/'fig_evidence_frames.pdf'); fig.savefig(OUT/'fig_evidence_frames.png',dpi=300); plt.close(fig)

print('generated', len(list(OUT.glob('*.pdf'))), 'PDF figures in', OUT)

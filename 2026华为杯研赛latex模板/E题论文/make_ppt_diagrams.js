const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Multimodal Emotion Recognition Paper';
pptx.subject = 'Auditable multimodal emotion recognition diagrams';
pptx.title = 'Unified timeline and hierarchical fusion diagrams';
pptx.company = 'Research modeling project';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN',
};
pptx.defineSlideMaster({
  title: 'WHITE',
  background: { color: 'FFFFFF' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.08, fill: { color: '24548E' }, line: { color: '24548E' } } },
    { text: { text: '复杂场景下的多模态情感识别', options: { x: 0.45, y: 7.12, w: 6.5, h: 0.18, fontFace: 'Microsoft YaHei', fontSize: 7, color: '505D70', margin: 0 } } },
  ],
  slideNumber: { x: 12.55, y: 7.10, color: '505D70', fontFace: 'Microsoft YaHei', fontSize: 7 },
});

const C = {
  blue: '24548E',
  teal: '087C70',
  orange: 'BA5336',
  gold: 'A17022',
  purple: '69509A',
  ink: '16243A',
  gray: '505D70',
  paleBlue: 'DFEAF3',
  paleTeal: 'DDEFE9',
  paleOrange: 'F7E6DF',
  paleGold: 'F6ECD8',
  palePurple: 'EAE3F3',
  paleGray: 'F2F5F8',
  white: 'FFFFFF',
};

const S = pptx.ShapeType;
const baseText = { fontFace: 'Microsoft YaHei', color: C.ink, margin: 0, breakLine: false, fit: 'shrink' };

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, { ...baseText, x, y, w, h, ...opts });
}

function addArrow(slide, x1, y1, x2, y2, color = C.gray, width = 1.6) {
  slide.addShape(S.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, beginArrowType: 'none', endArrowType: 'triangle' },
  });
}

function addDocumentIcon(slide, x, y, color) {
  slide.addShape(S.rect, { x, y, w: 0.42, h: 0.52, rectRadius: 0.03, fill: { color: C.white }, line: { color, width: 1.3 } });
  slide.addShape(S.line, { x: x + 0.08, y: y + 0.18, w: 0.25, h: 0, line: { color, width: 1.0 } });
  slide.addShape(S.line, { x: x + 0.08, y: y + 0.28, w: 0.25, h: 0, line: { color, width: 1.0 } });
  slide.addShape(S.line, { x: x + 0.08, y: y + 0.38, w: 0.16, h: 0, line: { color, width: 1.0 } });
}

function addMicIcon(slide, x, y, color) {
  slide.addShape(S.roundRect, { x: x + 0.15, y, w: 0.16, h: 0.34, rectRadius: 0.08, fill: { color: C.white }, line: { color, width: 1.2 } });
  slide.addShape(S.arc, { x: x + 0.06, y: y + 0.16, w: 0.34, h: 0.25, line: { color, width: 1.2 } });
  slide.addShape(S.line, { x: x + 0.23, y: y + 0.40, w: 0, h: 0.12, line: { color, width: 1.2 } });
  slide.addShape(S.line, { x: x + 0.12, y: y + 0.52, w: 0.22, h: 0, line: { color, width: 1.2 } });
}

function addCameraIcon(slide, x, y, color) {
  slide.addShape(S.roundRect, { x, y: y + 0.10, w: 0.50, h: 0.32, rectRadius: 0.04, fill: { color: C.white }, line: { color, width: 1.2 } });
  slide.addShape(S.triangle, { x: x + 0.44, y: y + 0.16, w: 0.16, h: 0.20, rotate: 90, fill: { color: C.white }, line: { color, width: 1.2 } });
  slide.addShape(S.ellipse, { x: x + 0.18, y: y + 0.16, w: 0.18, h: 0.18, fill: { color: C.paleBlue }, line: { color, width: 1.2 } });
}

function addBarsIcon(slide, x, y, color) {
  [0.16, 0.29, 0.43].forEach((h, i) => slide.addShape(S.rect, { x: x + i * 0.12, y: y + 0.52 - h, w: 0.07, h, fill: { color }, line: { color } }));
  slide.addShape(S.line, { x, y: y + 0.54, w: 0.34, h: 0, line: { color, width: 1.0 } });
}

function addTimelineIcon(slide, x, y, color) {
  slide.addShape(S.line, { x, y: y + 0.28, w: 0.62, h: 0, line: { color, width: 1.6, endArrowType: 'triangle' } });
  [0.10, 0.25, 0.40, 0.55].forEach((dx) => slide.addShape(S.ellipse, { x: x + dx, y: y + 0.21, w: 0.14, h: 0.14, fill: { color: C.white }, line: { color, width: 1.0 } }));
}

function addShieldIcon(slide, x, y, color) {
  slide.addShape(S.hexagon, { x, y, w: 0.50, h: 0.56, fill: { color: C.white }, line: { color, width: 1.3 } });
  slide.addShape(S.line, { x: x + 0.13, y: y + 0.28, w: 0.10, h: 0.10, line: { color, width: 1.8 } });
  slide.addShape(S.line, { x: x + 0.23, y: y + 0.38, w: 0.17, h: -0.20, line: { color, width: 1.8 } });
}

function addNetworkIcon(slide, x, y, color) {
  const pts = [[x + 0.08, y + 0.12], [x + 0.40, y + 0.05], [x + 0.30, y + 0.42], [x + 0.02, y + 0.36]];
  [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]].forEach(([a, b]) => slide.addShape(S.line, { x: pts[a][0], y: pts[a][1], w: pts[b][0] - pts[a][0], h: pts[b][1] - pts[a][1], line: { color, width: 1.0 } }));
  pts.forEach(([px, py]) => slide.addShape(S.ellipse, { x: px - 0.04, y: py - 0.04, w: 0.10, h: 0.10, fill: { color: C.white }, line: { color, width: 1.3 } }));
}

function addChartIcon(slide, x, y, color) {
  slide.addShape(S.line, { x, y: y + 0.55, w: 0.58, h: 0, line: { color, width: 1.0 } });
  slide.addShape(S.line, { x, y: y + 0.55, w: 0, h: -0.52, line: { color, width: 1.0 } });
  slide.addShape(S.line, { x: x + 0.08, y: y + 0.42, w: 0.12, h: -0.12, line: { color, width: 2.0 } });
  slide.addShape(S.line, { x: x + 0.20, y: y + 0.30, w: 0.14, h: 0.06, line: { color, width: 2.0 } });
  slide.addShape(S.line, { x: x + 0.34, y: y + 0.36, w: 0.14, h: -0.22, line: { color, width: 2.0 } });
}

function addNode(slide, x, y, w, h, fill, title, subtitle, icon, color) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color, width: 1.3 }, shadow: { type: 'outer', color: 'B8C2CC', blur: 1, angle: 45, distance: 1, opacity: 0.18 } });
  icon(slide, x + w / 2 - 0.25, y + 0.20, color);
  addText(slide, title, x + 0.10, y + 0.82, w - 0.20, 0.28, { fontSize: 14, bold: true, color, align: 'center' });
  addText(slide, subtitle, x + 0.08, y + 1.12, w - 0.16, 0.35, { fontSize: 8.5, color: C.gray, align: 'center', breakLine: true });
}

// Slide 1: end-to-end pipeline with visual icons.
{
  const slide = pptx.addSlide('WHITE');
  addText(slide, '统一时间轴与可审计多模态流程', 0.55, 0.30, 7.8, 0.42, { fontSize: 24, bold: true, color: C.blue });
  addText(slide, '从原始媒体到预测证据：每一层都有输入、质量检查和可复核输出', 0.58, 0.78, 8.7, 0.25, { fontSize: 11, color: C.gray });

  const nodes = [
    { x: 0.50, fill: C.paleBlue, color: C.blue, title: '原始媒体', subtitle: '文本 · 语音 · 视频', icon: (s, x, y, c) => { addDocumentIcon(s, x - 0.25, y + 0.04, c); addMicIcon(s, x + 0.04, y, c); addCameraIcon(s, x + 0.38, y + 0.02, c); } },
    { x: 3.05, fill: C.paleTeal, color: C.teal, title: '特征提取', subtitle: '768 / 74 / 35维', icon: (s, x, y, c) => { addBarsIcon(s, x - 0.17, y + 0.02, c); addDocumentIcon(s, x + 0.24, y + 0.01, c); } },
    { x: 5.60, fill: C.paleGold, color: C.gold, title: '公共时间轴', subtitle: '50个等宽窗口', icon: (s, x, y, c) => addTimelineIcon(s, x - 0.30, y + 0.02, c) },
    { x: 8.15, fill: C.palePurple, color: C.purple, title: '掩码与归一化', subtitle: '仅保留有效位置', icon: (s, x, y, c) => { addShieldIcon(s, x - 0.25, y, c); addBarsIcon(s, x + 0.15, y + 0.02, c); } },
    { x: 10.70, fill: C.paleOrange, color: C.orange, title: '预测与证据', subtitle: '类别 · 强度 · 追踪', icon: (s, x, y, c) => { addChartIcon(s, x - 0.30, y + 0.01, c); addNetworkIcon(s, x + 0.18, y + 0.03, c); } },
  ];
  nodes.forEach((n) => addNode(slide, n.x, 1.55, 2.10, 1.72, n.fill, n.title, n.subtitle, n.icon, n.color));
  nodes.slice(0, -1).forEach((n, i) => addArrow(slide, n.x + 2.15, 2.40, nodes[i + 1].x - 0.08, 2.40, C.gray, 1.7));

  const cards = [
    { x: 0.62, color: C.blue, icon: addShieldIcon, title: '质量审计', body: '覆盖率与时间单调性\n窗口不交叠与掩码一致性' },
    { x: 4.55, color: C.teal, icon: addNetworkIcon, title: '层次融合', body: '三路双向时序编码\n时间注意力与模态门控' },
    { x: 8.48, color: C.orange, icon: addChartIcon, title: '解释追踪', body: '模态消融与窗口遮挡\n关键帧与证据清单' },
  ];
  cards.forEach((card) => {
    slide.addShape(S.roundRect, { x: card.x, y: 4.10, w: 3.55, h: 1.42, rectRadius: 0.06, fill: { color: C.paleGray }, line: { color: 'E5E7EB', width: 0.8 } });
    card.icon(slide, card.x + 0.23, 4.46, card.color);
    addText(slide, card.title, card.x + 0.90, 4.30, 2.35, 0.24, { fontSize: 13, bold: true, color: card.color });
    addText(slide, card.body, card.x + 0.90, 4.66, 2.42, 0.52, { fontSize: 9.3, color: C.ink, breakLine: true });
  });
  addArrow(slide, 6.65, 3.28, 6.65, 4.02, C.gray, 1.4);
  addText(slide, '质量审计贯穿预处理、预测与解释全过程', 2.55, 6.10, 8.2, 0.28, { fontSize: 12, color: C.blue, align: 'center' });
}

// Slide 2: model architecture with three modality lanes and visual components.
{
  const slide = pptx.addSlide('WHITE');
  addText(slide, '层次化多模态情感预测模型', 0.55, 0.30, 7.2, 0.42, { fontSize: 24, bold: true, color: C.blue });
  addText(slide, '三路时序编码 → 模态内证据 → 样本级可靠性 → 双任务输出', 0.58, 0.78, 8.4, 0.25, { fontSize: 11, color: C.gray });

  const lanes = [
    { y: 1.50, color: C.orange, fill: C.paleOrange, title: '文本', sub: '768维', icon: addDocumentIcon },
    { y: 2.55, color: C.teal, fill: C.paleTeal, title: '语音', sub: '74维', icon: addMicIcon },
    { y: 3.60, color: C.blue, fill: C.paleBlue, title: '视觉', sub: '35维', icon: addCameraIcon },
  ];
  lanes.forEach((lane) => {
    slide.addShape(S.roundRect, { x: 0.55, y: lane.y, w: 1.35, h: 0.72, rectRadius: 0.06, fill: { color: lane.fill }, line: { color: lane.color, width: 1.1 } });
    lane.icon(slide, 0.73, lane.y + 0.08, lane.color);
    addText(slide, lane.title, 1.30, lane.y + 0.14, 0.47, 0.20, { fontSize: 11, bold: true, color: lane.color, align: 'center' });
    addText(slide, lane.sub, 1.30, lane.y + 0.39, 0.47, 0.15, { fontSize: 8.1, color: C.gray, align: 'center' });
    addArrow(slide, 1.98, lane.y + 0.36, 2.45, lane.y + 0.36, lane.color, 1.2);
    slide.addShape(S.roundRect, { x: 2.48, y: lane.y, w: 1.65, h: 0.72, rectRadius: 0.06, fill: { color: C.white }, line: { color: lane.color, width: 1.1 } });
    addNetworkIcon(slide, 2.66, lane.y + 0.10, lane.color);
    addText(slide, '双向循环编码', 3.25, lane.y + 0.16, 0.78, 0.19, { fontSize: 9.0, bold: true, color: lane.color, align: 'center' });
    addText(slide, '两层掩码序列', 3.25, lane.y + 0.41, 0.78, 0.14, { fontSize: 7.2, color: C.gray, align: 'center' });
    addArrow(slide, 4.17, lane.y + 0.36, 4.64, lane.y + 0.36, lane.color, 1.2);
    slide.addShape(S.roundRect, { x: 4.67, y: lane.y, w: 1.60, h: 0.72, rectRadius: 0.06, fill: { color: C.white }, line: { color: lane.color, width: 1.1 } });
    addTimelineIcon(slide, 4.98, lane.y + 0.06, lane.color);
    addText(slide, '时间注意力', 4.97, lane.y + 0.47, 1.02, 0.14, { fontSize: 8.2, bold: true, color: lane.color, align: 'center' });
  });

  addArrow(slide, 6.55, 1.86, 6.98, 3.02, C.gray, 1.2);
  addArrow(slide, 6.55, 2.91, 6.98, 3.02, C.gray, 1.2);
  addArrow(slide, 6.55, 3.96, 6.98, 3.02, C.gray, 1.2);
  slide.addShape(S.roundRect, { x: 7.02, y: 2.50, w: 1.58, h: 1.04, rectRadius: 0.08, fill: { color: C.paleGold }, line: { color: C.gold, width: 1.2 } });
  addNetworkIcon(slide, 7.54, 2.55, C.gold);
  addText(slide, '模态门控', 7.20, 3.11, 1.22, 0.18, { fontSize: 9.4, bold: true, color: C.ink, align: 'center' });
  addText(slide, '样本级可靠性', 7.15, 3.37, 1.30, 0.12, { fontSize: 7.1, color: C.gray, align: 'center' });
  addArrow(slide, 8.65, 3.02, 9.04, 3.02, C.gray, 1.5);
  slide.addShape(S.roundRect, { x: 9.08, y: 2.50, w: 1.56, h: 1.04, rectRadius: 0.08, fill: { color: C.palePurple }, line: { color: C.purple, width: 1.2 } });
  addBarsIcon(slide, 9.72, 2.54, C.purple);
  addText(slide, '特征融合', 9.25, 3.11, 1.22, 0.18, { fontSize: 9.4, bold: true, color: C.ink, align: 'center' });
  addText(slide, '归一化与非线性变换', 9.22, 3.37, 1.28, 0.12, { fontSize: 7.1, color: C.gray, align: 'center' });
  addArrow(slide, 10.70, 3.02, 11.02, 3.02, C.gray, 1.5);

  slide.addShape(S.roundRect, { x: 11.06, y: 1.80, w: 1.66, h: 1.06, rectRadius: 0.08, fill: { color: C.paleBlue }, line: { color: C.blue, width: 1.2 } });
  addChartIcon(slide, 11.28, 1.99, C.blue);
  addText(slide, '情感极性', 11.91, 2.02, 0.67, 0.18, { fontSize: 9.5, bold: true, color: C.blue, align: 'center' });
  addText(slide, '三类别', 11.91, 2.29, 0.67, 0.14, { fontSize: 7.8, color: C.gray, align: 'center' });
  slide.addShape(S.roundRect, { x: 11.06, y: 3.18, w: 1.66, h: 1.06, rectRadius: 0.08, fill: { color: C.paleOrange }, line: { color: C.orange, width: 1.2 } });
  addChartIcon(slide, 11.28, 3.37, C.orange);
  addText(slide, '情感强度', 11.91, 3.40, 0.67, 0.18, { fontSize: 9.5, bold: true, color: C.orange, align: 'center' });
  addText(slide, '连续数值', 11.91, 3.67, 0.67, 0.14, { fontSize: 7.8, color: C.gray, align: 'center' });
  addArrow(slide, 10.70, 3.02, 11.02, 2.33, C.gray, 1.2);
  addArrow(slide, 10.70, 3.02, 11.02, 3.70, C.gray, 1.2);

  slide.addShape(S.roundRect, { x: 2.45, y: 5.10, w: 8.90, h: 0.78, rectRadius: 0.06, fill: { color: C.paleGray }, line: { color: 'D1D5DB', width: 0.8 } });
  addShieldIcon(slide, 2.73, 5.21, C.blue);
  addText(slide, '缺失窗口掩码', 3.36, 5.20, 1.55, 0.18, { fontSize: 10.5, bold: true, color: C.blue });
  addText(slide, '缺失值归零 + 显式有效位置 + 注意力分母掩码', 5.05, 5.20, 5.78, 0.22, { fontSize: 9.2, color: C.ink });
  addText(slide, '解释输出：模态消融 · 注意力排序 · 局部窗口遮挡 · 关键帧追踪', 1.40, 6.24, 10.5, 0.25, { fontSize: 11, color: C.orange, align: 'center' });
}

pptx.writeFile({ fileName: 'figures/ppt_diagrams.pptx' });

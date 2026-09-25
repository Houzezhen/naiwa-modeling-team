// Editable PowerPoint source for Figure 1.1: the relationship among Q1–Q3.
// The final PDF/PNG is exported from this exact slide by PowerPoint.
const pptxgen = require('pptxgenjs');
const path = require('path');

const pptx = new pptxgen();
pptx.defineLayout({ name: 'PAPER_ROADMAP', width: 10, height: 8.2 });
pptx.layout = 'PAPER_ROADMAP';
pptx.author = 'Math Modeling Team';
pptx.subject = 'Three-problem multimodal emotion modeling roadmap';
pptx.title = '三个问题的建模流程及数据关系';
pptx.lang = 'zh-CN';
pptx.theme = { headFontFace: 'Microsoft YaHei', bodyFontFace: 'Microsoft YaHei', lang: 'zh-CN' };

const S = pptx.ShapeType;
const C = {
  ink: '203247', muted: '526071', line: '8799A9',
  blue: '255C8E', bluePale: 'EEF4F9',
  teal: '087C70', tealPale: 'EAF5F1',
  orange: 'AD5835', orangePale: 'FCF1EA',
  white: 'FFFFFF',
};
const slide = pptx.addSlide();
slide.background = { color: C.white };

function txt(value, x, y, w, h, opts = {}) {
  slide.addText(value, {
    x, y, w, h, fontFace: 'Microsoft YaHei', fontSize: 12,
    color: C.ink, margin: 0, valign: 'mid', breakLine: false,
    fit: 'shrink', ...opts,
  });
}
function border(x, y, w, h, color) {
  slide.addShape(S.rect, {
    x, y, w, h, rectRadius: 0,
    fill: { color: C.white, transparency: 100 },
    line: { color, width: 1.05, dashType: 'dash' },
  });
}
function box(x, y, w, h, fill, line, title, detail) {
  slide.addShape(S.rect, {
    x, y, w, h, rectRadius: 0,
    fill: { color: fill }, line: { color: line, width: 0.8 },
  });
  txt(title, x + .08, y + .13, w - .16, .34,
      { fontSize: 13.3, color: line, align: 'center' });
  txt(detail, x + .08, y + .52, w - .16, h - .62,
      { fontSize: 10.9, color: C.ink, align: 'center', valign: 'top', breakLine: true });
}
function arrow(x1, y1, x2, y2, color = C.line, width = 1.45) {
  slide.addShape(S.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, beginArrowType: 'none', endArrowType: 'triangle' },
  });
}
function lane(y, title, subtitle, color, pale, nodes) {
  border(.25, y, 9.5, 2.08, color);
  txt(title, .44, y + .22, .88, .42,
      { fontSize: 15.5, color, align: 'center' });
  txt(subtitle, .43, y + .70, .90, .70,
      { fontSize: 10.8, color: C.muted, align: 'center', breakLine: true, valign: 'top' });
  const xs = [1.48, 3.57, 5.66, 7.75];
  nodes.forEach((n, i) => box(xs[i], y + .47, 1.73, 1.12, pale, color, n[0], n[1]));
  for (let i = 0; i < 3; i++) arrow(xs[i] + 1.77, y + 1.03, xs[i + 1] - .08, y + 1.03, color);
}

lane(.45, '问题一', '原始数据与\n时间对齐', C.blue, C.bluePale, [
  ['附件1原始视频', '100条视频\n转写与标注'],
  ['三模态特征', '文本 · 语音 · 视觉\n保留样本ID'],
  ['公共时间轴', '50个时间窗\n显式有效掩码'],
  ['质量核验', '覆盖率 · 边界\n原始片段回看'],
]);
lane(3.02, '问题二', '缺失条件下\n鲁棒预测', C.teal, C.tealPale, [
  ['附件2特征', '训练 / 验证 / 测试\n统一特征版本'],
  ['掩码归一化', '只用训练集统计量\n保留缺失位置'],
  ['时序融合', '双向GRU\n时间注意力 · 门控'],
  ['预测与检验', '极性 + 强度\n遮挡实验 · 附件3'],
]);
lane(5.58, '问题三', '模型解释与\n证据追踪', C.orange, C.orangePale, [
  ['附件4样本', '无标签特征\n配套原始视频'],
  ['统一主模型', '复用附件2选定模型\n输出极性与强度'],
  ['作用量化', '模态消融\n时间窗遮挡'],
  ['证据定位', '主要模态 · 时间段\n文本/语音/关键帧'],
]);

// Inter-problem links transfer modeling rules and the selected checkpoint,
// not training samples.  The two input attachments stay in separate lanes.
arrow(6.53, 2.54, 6.53, 2.95, C.blue, 1.4);
txt('对齐与掩码规则', 6.71, 2.63, 1.78, .25,
    { fontSize: 10.2, color: C.blue });
arrow(4.43, 5.11, 4.43, 5.52, C.teal, 1.4);
txt('主模型与输入接口', 4.62, 5.20, 1.90, .25,
    { fontSize: 10.2, color: C.teal });
txt('附件1与附件2是不同样本集合；附件3、附件4只用于专项推理。', .36, 7.86, 9.27, .20,
    { fontSize: 9.9, color: C.muted, align: 'center' });

function miniDeck(fileName, tag, color, pale, items, footer) {
  const deck = new pptxgen();
  deck.defineLayout({ name: 'ANALYSIS', width: 10, height: 2.52 });
  deck.layout = 'ANALYSIS';
  deck.author = 'Math Modeling Team';
  deck.lang = 'zh-CN';
  deck.theme = { headFontFace: 'Microsoft YaHei', bodyFontFace: 'Microsoft YaHei', lang: 'zh-CN' };
  const page = deck.addSlide();
  page.background = { color: C.white };
  page.addShape(S.rect, {
    x: .20, y: .28, w: 9.60, h: 1.90,
    fill: { color: C.white, transparency: 100 },
    line: { color, width: 1.05, dashType: 'dash' },
  });
  page.addShape(S.roundRect, {
    x: .38, y: .12, w: 1.17, h: .37, rectRadius: .06,
    fill: { color }, line: { color, transparency: 100 },
  });
  page.addText(tag, {
    x: .43, y: .17, w: 1.06, h: .22, margin: 0,
    fontFace: 'Microsoft YaHei', fontSize: 11.3,
    color: C.white, align: 'center', valign: 'mid',
  });
  const xs = [.42, 2.91, 5.40, 7.89];
  items.forEach((item, i) => {
    page.addShape(S.roundRect, {
      x: xs[i], y: .72, w: 1.69, h: .99, rectRadius: .08,
      fill: { color: pale }, line: { color, width: 1.0 },
    });
    page.addText(item[0], {
      x: xs[i] + .07, y: .88, w: 1.55, h: .29,
      margin: 0, fontFace: 'Microsoft YaHei', fontSize: 12.0,
      color, align: 'center', valign: 'mid', fit: 'shrink',
    });
    page.addText(item[1], {
      x: xs[i] + .06, y: 1.21, w: 1.57, h: .36,
      margin: 0, fontFace: 'Microsoft YaHei', fontSize: 9.5,
      color: C.ink, align: 'center', valign: 'mid',
      fit: 'shrink', breakLine: true,
    });
    if (i < 3) {
      page.addShape(S.line, {
        x: xs[i] + 1.78, y: 1.21, w: .62, h: 0,
        line: { color, width: 1.65, beginArrowType: 'none', endArrowType: 'triangle' },
      });
    }
  });
  page.addText(footer, {
    x: .43, y: 1.87, w: 9.13, h: .20,
    margin: 0, fontFace: 'Microsoft YaHei', fontSize: 9.2,
    color: C.muted, align: 'center', valign: 'mid',
  });
  return deck.writeFile({ fileName: path.join(__dirname, 'figures', fileName) });
}

async function writeAll() {
  await pptx.writeFile({ fileName: path.join(__dirname, 'figures', 'fig_three_question_roadmap.pptx') });
  await miniDeck('fig_problem1_analysis.pptx', '问题一', C.blue, C.bluePale, [
    ['原始数据', '附件1视频、转写与标注'],
    ['三模态特征', '文本768 · 语音74 · 视觉35'],
    ['公共时间轴', '50窗映射与有效掩码'],
    ['对齐核验', '覆盖率、边界与视频回看'],
  ], '输出：可追溯的窗口特征、掩码、时间索引与质量记录');
  await miniDeck('fig_problem2_analysis.pptx', '问题二', C.teal, C.tealPale, [
    ['统一输入', '附件2划分与特征版本'],
    ['掩码处理', '训练集归一化 · 缺失位置'],
    ['融合与双任务', '双向GRU · 注意力 · 门控'],
    ['性能检验', '验证选模 · 遮挡 · 附件3'],
  ], '输出：情感极性、连续强度及局部缺失敏感性');
  await miniDeck('fig_problem3_analysis.pptx', '问题三', C.orange, C.orangePale, [
    ['选定主模型', '复用附件2验证选模结果'],
    ['模态作用', '门控权重与反事实消融'],
    ['局部证据', '时间窗遮挡与原始时刻'],
    ['预测交付', '附件4极性、强度、证据'],
  ], '输出：主要参考模态、重要时间段与可回看的证据');
}
writeAll().catch(err => { console.error(err); process.exitCode = 1; });

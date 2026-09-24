const fs = require('fs');

const OUT = 'figures';
const C = { blue: '#24548E', teal: '#087C70', orange: '#BA5336', gold: '#A17022', purple: '#69509A', ink: '#16243A', gray: '#505D70', paleBlue: '#DFEAF3', paleTeal: '#DDEFE9', paleOrange: '#F7E6DF', paleGold: '#F6ECD8', palePurple: '#EAE3F3', paleGray: '#F2F5F8' };

function esc(value) { return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function text(x, y, value, size = 18, color = C.ink, weight = 400, anchor = 'start', italic = false) {
  return `<text x="${x}" y="${y}" font-family="Microsoft YaHei, SimHei" font-size="${size}px" font-weight="${weight}" fill="${color}" text-anchor="${anchor}"${italic ? ' font-style="italic"' : ''}>${esc(value)}</text>`;
}
function rect(x, y, w, h, fill, stroke = '#D1D5DB', r = 14, sw = 2) { return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`; }
function line(x1, y1, x2, y2, stroke = C.gray, sw = 2, arrow = false) { return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}"${arrow ? ' marker-end="url(#arrow)"' : ''}/>`; }
function circle(cx, cy, r, fill = '#fff', stroke = C.blue, sw = 2) { return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`; }
function header(title, subtitle) { return `<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="${C.gray}"/></marker></defs><rect width="1280" height="720" fill="#FFFFFF"/><rect width="1280" height="8" fill="${C.blue}"/>${text(54, 54, title, 30, C.blue, 700)}${text(56, 86, subtitle, 14, C.gray)}`; }
function footer() { return `${text(52, 692, '复杂场景下的多模态情感识别', 10, C.gray)}${text(1230, 692, '可编辑框图对应的矢量图', 10, C.gray, 400, 'end')}`; }

function documentIcon(x, y, color) { return `${rect(x, y, 38, 48, '#fff', color, 4, 2)}${line(x + 8, y + 17, x + 30, y + 17, color, 1.5)}${line(x + 8, y + 27, x + 30, y + 27, color, 1.5)}${line(x + 8, y + 37, x + 23, y + 37, color, 1.5)}`; }
function micIcon(x, y, color) { return `<rect x="${x + 13}" y="${y}" width="14" height="32" rx="7" fill="#fff" stroke="${color}" stroke-width="2"/><path d="M${x + 5},${y + 18} A16,16 0 0 0 ${x + 37},${y + 18}" fill="none" stroke="${color}" stroke-width="2"/><line x1="${x + 21}" y1="${y + 34}" x2="${x + 21}" y2="${y + 46}" stroke="${color}" stroke-width="2"/><line x1="${x + 10}" y1="${y + 46}" x2="${x + 32}" y2="${y + 46}" stroke="${color}" stroke-width="2"/>`; }
function cameraIcon(x, y, color) { return `<rect x="${x}" y="${y + 10}" width="44" height="27" rx="5" fill="#fff" stroke="${color}" stroke-width="2"/><path d="M${x + 42},${y + 15} L${x + 56},${y + 21} L${x + 56},${y + 32} L${x + 42},${y + 37} Z" fill="#fff" stroke="${color}" stroke-width="2"/>${circle(x + 20, y + 23, 7, C.paleBlue, color, 2)}`; }
function barsIcon(x, y, color) { return `${line(x, y + 50, x + 52, y + 50, color, 2)}${[15, 30, 46].map((h, i) => `<rect x="${x + 8 + i * 14}" y="${y + 50 - h}" width="8" height="${h}" fill="${color}"/>`).join('')}`; }
function timelineIcon(x, y, color) { return `${line(x, y + 28, x + 64, y + 28, color, 2, true)}${[10, 25, 40, 55].map(dx => circle(x + dx, y + 28, 7, '#fff', color, 2)).join('')}`; }
function shieldIcon(x, y, color) { return `<path d="M${x + 25},${y} L${x + 47},${y + 9} L${x + 43},${y + 38} L${x + 25},${y + 53} L${x + 7},${y + 38} L${x + 3},${y + 9} Z" fill="#fff" stroke="${color}" stroke-width="2"/>${line(x + 14, y + 27, x + 22, y + 35, color, 3)}${line(x + 22, y + 35, x + 38, y + 17, color, 3)}`; }
function networkIcon(x, y, color) { const p = [[x + 8, y + 10], [x + 42, y + 3], [x + 32, y + 43], [x + 2, y + 36]]; const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]; return edges.map(([a, b]) => line(p[a][0], p[a][1], p[b][0], p[b][1], color, 1.5)).join('') + p.map(([px, py]) => circle(px, py, 6, '#fff', color, 2)).join(''); }
function chartIcon(x, y, color) { return `${line(x, y + 54, x + 60, y + 54, color, 2)}${line(x, y + 54, x, y, color, 2)}${line(x + 10, y + 42, x + 24, y + 28, color, 3)}${line(x + 24, y + 28, x + 38, y + 35, color, 3)}${line(x + 38, y + 35, x + 54, y + 8, color, 3)}`; }

function node(x, fill, title, subtitle, icon, color) {
  return rect(x, 156, 202, 164, fill, color) + icon(x + 77, 174, color) + text(x + 101, 260, title, 18, color, 700, 'middle') + text(x + 101, 285, subtitle, 12, C.gray, 400, 'middle');
}

let pipeline = `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">${header('统一时间轴与可审计多模态流程', '从原始媒体到预测证据：每一层都有输入、质量检查和可复核输出')}`;
pipeline += node(48, C.paleBlue, '原始媒体', '文本 · 语音 · 视频', (x, y, c) => documentIcon(x - 28, y + 4, c) + micIcon(x + 13, y, c) + cameraIcon(x + 48, y + 4, c), C.blue);
pipeline += node(304, C.paleTeal, '特征提取', '768 / 74 / 35维', (x, y, c) => barsIcon(x - 20, y + 3, c) + documentIcon(x + 25, y + 2, c), C.teal);
pipeline += node(560, C.paleGold, '公共时间轴', '50个等宽窗口', (x, y, c) => timelineIcon(x - 33, y + 4, c), C.gold);
pipeline += node(816, C.palePurple, '掩码与归一化', '仅保留有效位置', (x, y, c) => shieldIcon(x - 28, y, c) + barsIcon(x + 20, y + 2, c), C.purple);
pipeline += node(1072, C.paleOrange, '预测与证据', '类别 · 强度 · 追踪', (x, y, c) => chartIcon(x - 31, y, c) + networkIcon(x + 26, y + 3, c), C.orange);
pipeline += [[250, 238, 296, 238], [506, 238, 552, 238], [762, 238, 808, 238], [1018, 238, 1064, 238]].map(a => line(...a, C.gray, 2, true)).join('');
const cards = [[62, C.blue, '质量审计', '覆盖率与时间单调性\n窗口不交叠与掩码一致性', shieldIcon], [455, C.teal, '层次融合', '三路双向时序编码\n时间注意力与模态门控', networkIcon], [848, C.orange, '解释追踪', '模态消融与窗口遮挡\n关键帧与证据清单', chartIcon]];
cards.forEach(([x, color, title, body, icon]) => { pipeline += rect(x, 410, 355, 142, C.paleGray, '#E5E7EB', 10, 1) + icon(x + 24, 445, color) + text(x + 90, 449, title, 17, color, 700) + text(x + 90, 482, body.split('\n')[0], 13, C.ink) + text(x + 90, 506, body.split('\n')[1], 13, C.ink); });
pipeline += line(640, 324, 640, 402, C.gray, 2, true) + text(640, 625, '质量审计贯穿预处理、预测与解释全过程', 16, C.blue, 400, 'middle') + footer() + '</svg>';
fs.writeFileSync(`${OUT}/fig_pipeline_ppt.svg`, pipeline, 'utf8');

let architecture = `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">${header('层次化多模态情感预测模型', '三路时序编码 → 模态内证据 → 样本级可靠性 → 双任务输出')}`;
const lanes = [[150, C.orange, C.paleOrange, '文本', '768维', documentIcon], [255, C.teal, C.paleTeal, '语音', '74维', micIcon], [360, C.blue, C.paleBlue, '视觉', '35维', cameraIcon]];
lanes.forEach(([y, color, fill, title, sub, icon]) => {
  architecture += rect(52, y, 135, 72, fill, color, 10, 2) + icon(70, y + 12, color) + text(157, y + 27, title, 14, color, 700, 'middle') + text(157, y + 50, sub, 10, C.gray, 400, 'middle');
  architecture += line(198, y + 36, 244, y + 36, color, 2, true) + rect(248, y, 165, 72, '#fff', color, 10, 2) + networkIcon(270, y + 13, color) + text(365, y + 29, '双向循环编码', 11, color, 700, 'middle') + text(365, y + 51, '两层掩码序列', 9, C.gray, 400, 'middle');
  architecture += line(424, y + 36, 468, y + 36, color, 2, true) + rect(472, y, 160, 72, '#fff', color, 10, 2) + timelineIcon(516, y + 7, color) + text(552, y + 58, '时间注意力', 10, color, 700, 'middle');
});
architecture += line(640, 186, 695, 302, C.gray, 2, true) + line(640, 291, 695, 302, C.gray, 2, true) + line(640, 396, 695, 302, C.gray, 2, true);
architecture += rect(700, 250, 158, 104, C.paleGold, C.gold, 12, 2) + networkIcon(750, 256, C.gold) + text(779, 327, '模态门控', 13, C.ink, 700, 'middle') + text(779, 346, '样本级可靠性', 9, C.gray, 400, 'middle');
architecture += line(865, 302, 902, 302, C.gray, 2, true) + rect(906, 250, 156, 104, C.palePurple, C.purple, 12, 2) + barsIcon(966, 254, C.purple) + text(984, 327, '特征融合', 13, C.ink, 700, 'middle') + text(984, 346, '归一化与非线性变换', 9, C.gray, 400, 'middle');
architecture += line(1068, 302, 1100, 232, C.gray, 2, true) + line(1068, 302, 1100, 370, C.gray, 2, true);
architecture += rect(1104, 180, 165, 106, C.paleBlue, C.blue, 12, 2) + chartIcon(1124, 201, C.blue) + text(1220, 226, '情感极性', 12, C.blue, 700, 'middle') + text(1220, 250, '三类别', 9, C.gray, 400, 'middle');
architecture += rect(1104, 318, 165, 106, C.paleOrange, C.orange, 12, 2) + chartIcon(1124, 339, C.orange) + text(1220, 364, '情感强度', 12, C.orange, 700, 'middle') + text(1220, 388, '连续数值', 9, C.gray, 400, 'middle');
architecture += rect(246, 510, 850, 78, C.paleGray, '#D1D5DB', 10, 1) + shieldIcon(270, 522, C.blue) + text(350, 546, '缺失窗口掩码', 14, C.blue, 700) + text(552, 546, '缺失值归零 + 显式有效位置 + 注意力分母掩码', 13, C.ink) + text(640, 645, '解释输出：模态消融 · 注意力排序 · 局部窗口遮挡 · 关键帧追踪', 15, C.orange, 400, 'middle') + footer() + '</svg>';
fs.writeFileSync(`${OUT}/fig_model_architecture_ppt.svg`, architecture, 'utf8');

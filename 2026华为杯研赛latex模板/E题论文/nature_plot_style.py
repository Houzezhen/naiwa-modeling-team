"""Shared Chinese typography and restrained manuscript-figure palette."""

import matplotlib as mpl


INK = '#16243A'
GRAY = '#505D70'
BLUE = '#24548E'
TEAL = '#087C70'
ORANGE = '#BA5336'
GOLD = '#A17022'
PURPLE = '#69509A'
PALE = '#E8EFF5'
GRID = '#DFE6EB'
CHINESE_CLASSES = {'Negative': '消极', 'Neutral': '中性', 'Positive': '积极'}
CHINESE_MODALITIES = {'text': '文本', 'audio': '语音', 'vision': '视觉'}


def apply_style():
    mpl.rcParams.update({
        'font.family': ['Times New Roman', 'SimHei'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.fallback': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.titleweight': 'semibold',
        'axes.edgecolor': GRAY,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': GRAY,
        'ytick.color': GRAY,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': .65,
        'grid.color': GRID,
        'grid.linewidth': .65,
        'legend.frameon': False,
        'axes.unicode_minus': False,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
    })

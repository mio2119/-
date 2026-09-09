"""Create original data figures and compact explanatory diagrams; no PDF output."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
PALETTE = ['#2E5C8A', '#D87836', '#42866B', '#8A647A']


def configure():
    """Prefer installed CJK fonts while preserving portability."""
    for path in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 10.5, 'axes.labelsize': 10.5,
        'axes.titlesize': 11, 'legend.fontsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'lines.linewidth': 1.8,
        'svg.fonttype': 'path', 'savefig.facecolor': 'white',
    })
    FIG.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def styled(ax, xlabel='波数（cm$^{-1}$）', ylabel='反射率（%）'):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.6)


def raw_spectra():
    for ids, name, title in [([1, 2], 'raw_sic', '碳化硅'), ([3, 4], 'raw_silicon', '硅')]:
        fig, ax = plt.subplots(figsize=(5.7, 3.25), layout='constrained')
        for k, i in enumerate(ids):
            a = np.loadtxt(ROOT / 'data' / f'spectrum_{i}.csv', delimiter=',', skiprows=1)
            ax.plot(a[1:, 0], a[1:, 1], color=PALETTE[k],
                    label=f'{10 if i % 2 else 15}°入射', lw=1.4)
        if name == 'raw_sic':
            ax.axvspan(2000, 3300, color=PALETTE[2], alpha=0.10)
            ax.axvspan(2300, 2400, color='#999999', alpha=0.15)
        ax.legend(loc='upper right', frameon=False)
        styled(ax)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(5.7, 3.25), layout='constrained')
    for k, i in enumerate([1, 2]):
        a = np.loadtxt(ROOT / 'data' / f'spectrum_{i}.csv', delimiter=',', skiprows=1)
        mask = (a[:, 0] >= 2000) & (a[:, 0] <= 3300)
        ax.plot(a[mask, 0], a[mask, 1], color=PALETTE[k], label=f'附件{i}')
    ax.axvspan(2300, 2400, color='#aaaaaa', alpha=0.2, label='局部屏蔽段')
    styled(ax)
    ax.legend(frameon=False)
    save(fig, 'sic_window')


def box(ax, xy, text, color, w=2.45, h=0.75):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.035,rounding_size=0.09',
                          linewidth=1.0, edgecolor=color, facecolor=color + '18')
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, va='center', ha='center', fontsize=10.5)


def arrow(ax, start, end, color='#59636B'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
                                linewidth=1.1, color=color))


def workflow():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.set(xlim=(0, 8.6), ylim=(-0.1, 4.25))
    ax.axis('off')
    entries = [
        ((0.15, 3.25), '原始光谱\n单位与异常核对', PALETTE[0]),
        ((3.07, 3.25), '条纹频率\n确定厚度初值', PALETTE[0]),
        ((5.99, 3.25), '文献色散\n核对适用波段', PALETTE[0]),
        ((0.15, 1.70), '两光束模型\n碳化硅双角拟合', PALETTE[2]),
        ((3.07, 1.70), '多光束模型\n硅双角拟合', PALETTE[2]),
        ((5.99, 1.70), '相同数据段\n比较残差与预测', PALETTE[2]),
        ((0.15, 0.15), '分角与分窗口\n一致性检验', PALETTE[1]),
        ((3.07, 0.15), '参数扰动\n条件不确定性', PALETTE[1]),
        ((5.99, 0.15), '厚度与适用边界\n代码和结果归档', PALETTE[1]),
    ]
    for e in entries:
        box(ax, *e)
    for y in [3.625, 2.075, 0.525]:
        arrow(ax, (2.65, y), (3.02, y))
        arrow(ax, (5.57, y), (5.94, y))
    for x in [1.375, 4.295, 7.215]:
        arrow(ax, (x, 3.20), (x, 2.50))
        arrow(ax, (x, 1.65), (x, 0.95))
    save(fig, 'workflow')


def geometry():
    fig, ax = plt.subplots(figsize=(5.7, 3.25))
    ax.set(xlim=(-0.2, 8), ylim=(-1.3, 4.8))
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 7.8, 2, color='#DCE9F2', zorder=0))
    ax.add_patch(Rectangle((0, -1.2), 7.8, 1.2, color='#E7E7E7', zorder=0))
    for y in [0, 2]:
        ax.plot([0, 7.8], [y, y], color='#56616B', lw=1.2)
    for start, end, c in [((0.8, 4.1), (2.0, 2), PALETTE[0]),
                          ((2.0, 2), (3.2, 4.1), PALETTE[0]),
                          ((2.0, 2), (2.7, 0), PALETTE[1]),
                          ((2.7, 0), (3.4, 2), PALETTE[1]),
                          ((3.4, 2), (4.6, 4.1), PALETTE[1]),
                          ((3.4, 2), (4.1, 0), PALETTE[2]),
                          ((4.1, 0), (4.8, 2), PALETTE[2]),
                          ((4.8, 2), (6.0, 4.1), PALETTE[2])]:
        arrow(ax, start, end, c)
    ax.plot([2, 2], [0.5, 4.2], '--', color='#777777', lw=0.7)
    ax.text(0.25, 4.35, '入射光', color=PALETTE[0])
    ax.text(3.0, 4.25, '表面反射', color=PALETTE[0])
    ax.text(4.4, 3.55, '一次往返', color=PALETTE[1])
    ax.text(5.9, 4.25, '高阶出射', color=PALETTE[2])
    ax.text(6.5, 2.7, '空气  $n_0$')
    ax.text(6.4, 1.0, '外延层  $n_1$')
    ax.text(6.5, -0.7, '衬底  $n_2$')
    ax.annotate('', xy=(0.45, 2), xytext=(0.45, 0),
                arrowprops={'arrowstyle': '<->', 'color': '#333333'})
    ax.text(0.17, 1.0, 'd', va='center', fontsize=12)
    ax.text(2.07, 3.0, r'$\theta$', fontsize=12)
    ax.text(2.04, 0.80, r'$\beta$', fontsize=12)
    save(fig, 'geometry')


def quality_table():
    q = json.loads((ROOT / 'results' / 'data_quality.json').read_text(encoding='utf-8'))
    rows = [r'\begin{table}[htbp]\centering', r'\caption{四个附件的数据核对结果}\label{tab:quality}',
            r'\begin{tabular}{cccccc}', r'\toprule 附件 & 材料 & 角度 & 观测数 & 最大反射率/\% & 超100\%点数\\\midrule']
    for s in q:
        rows.append(f"{s['id']} & {s['material']} & {s['angle_degree']}$^\\circ$ & {s['rows']} & {s['reflectance_max']:.4f} & {s['over100_count']} " + r'\\')
    rows.extend([r'\bottomrule', r'\end{tabular}\end{table}'])
    (ROOT / 'tables').mkdir(exist_ok=True)
    (ROOT / 'tables' / 'data_quality.tex').write_text('\n'.join(rows), encoding='utf-8')


def main():
    configure()
    raw_spectra()
    workflow()
    geometry()
    quality_table()


if __name__ == '__main__':
    main()

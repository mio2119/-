"""Render numerical evidence directly from solver outputs and make LaTeX tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_figures import configure, save, styled, PALETTE, ROOT


def table(filename, caption, label, headings, rows, spec=None):
    spec = spec or ('l' + 'r' * (len(headings) - 1))
    lines = [r'\begin{table}[htbp]\centering',
             '\\caption{' + caption + '}\\label{' + label + '}',
             r'\begin{tabular}{' + spec + '}', r'\toprule',
             ' & '.join(headings) + r'\\\midrule']
    lines.extend(' & '.join(map(str, row)) + r'\\' for row in rows)
    lines += [r'\bottomrule', r'\end{tabular}\end{table}']
    (ROOT / 'tables' / f'{filename}.tex').write_text('\n'.join(lines), encoding='utf-8')


def gap_plot(ax, x, y, **kwargs):
    """Split removed spectral bands instead of drawing through missing samples."""
    cuts = np.r_[0, np.flatnonzero(np.diff(x) > 5) + 1, len(x)]
    for j, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        kw = kwargs.copy()
        if j:
            kw.pop('label', None)
        ax.plot(x[lo:hi], y[lo:hi], **kw)


def sic_figures(s):
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 4.0), sharex=True, layout='constrained')
    for i, ax in enumerate(axes, 1):
        a = pd.read_csv(ROOT / 'results' / f'sic_prediction_{i}.csv')
        x = a.wavenumber_cm_inverse.to_numpy()
        gap_plot(ax, x, a.reflectance_percent.to_numpy(), color=PALETTE[0], lw=1.4, label='实测光谱')
        gap_plot(ax, x, a.two_beam_percent.to_numpy(), color=PALETTE[1], ls='--', label='色散两光束拟合')
        ax.text(.02, .86, f'（{chr(96+i)}）{10 if i==1 else 15}°', transform=ax.transAxes)
        styled(ax, xlabel='' if i == 1 else '波数（cm$^{-1}$）')
        ax.axvspan(2300, 2400, color='#aaaaaa', alpha=.12)
    axes[0].legend(frameon=False, loc='lower right', fontsize=8)
    save(fig, 'sic_fit')

    fig, ax = plt.subplots(figsize=(5.7, 3.2), layout='constrained')
    for i in [1, 2]:
        a = pd.read_csv(ROOT / 'results' / f'sic_fft_{i}.csv')
        ax.plot(a.frequency_cm * 1000, a.power / a.power.max(), color=PALETTE[i-1],
                ls='-' if i==1 else '--', label=f'附件{i}')
    styled(ax, '波数域频率（10$^{-3}$ cm）', '归一化功率（1）')
    ax.legend(frameon=False)
    save(fig, 'sic_fft')

    fig, ax = plt.subplots(figsize=(5.7, 3.2), layout='constrained')
    cv = s['cross_validation']['folds']
    x = np.arange(1, 6)
    ax.plot(x, [v['two_beam_rmse_pp'] for v in cv], 'o-', color=PALETTE[0], label='两光束')
    ax.plot(x, [v['airy_rmse_pp'] for v in cv], 's--', color=PALETTE[1], label='有效多光束核')
    styled(ax, '留出折次', '留出均方根误差（百分点）')
    ax.set_xticks(x)
    ax.legend(frameon=False)
    save(fig, 'sic_airy_cv')


def silicon_figures(s):
    a = np.load(ROOT / 'results' / 'silicon_predictions.npz')
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 4.0), sharex=True, layout='constrained')
    for i, ax in zip([3, 4], axes):
        x = a[f'sigma_{i}']
        ax.plot(x, a[f'R_{i}'], color=PALETTE[0], lw=1.2, label='实测光谱')
        ax.plot(x, a[f'multi_{i}'], color=PALETTE[1], ls='--', lw=1.2, label='多光束共同厚度')
        ax.text(.50, .80, f'（{chr(i+94)}）{10 if i==3 else 15}°', transform=ax.transAxes)
        styled(ax, xlabel='' if i==3 else '波数（cm$^{-1}$）')
    axes[0].legend(frameon=False, loc='upper right', fontsize=8)
    save(fig, 'silicon_fit')

    fig, axes = plt.subplots(2, 1, figsize=(5.7, 3.8), sharex=True, layout='constrained')
    for i, ax in zip([3, 4], axes):
        x = a[f'sigma_{i}']
        ax.plot(x, a[f'residual_two_{i}'], color=PALETTE[1], alpha=.7, lw=1.2, label='两光束残差')
        ax.plot(x, a[f'residual_multi_{i}'], color=PALETTE[0], lw=1.2, label='多光束残差')
        ax.axhline(0, color='#666666', lw=.6)
        ax.text(.80, .79, f'{10 if i==3 else 15}°入射', transform=ax.transAxes)
        styled(ax, xlabel='' if i==3 else '波数（cm$^{-1}$）', ylabel='残差（百分点）')
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, 'silicon_residuals')

    fig, ax = plt.subplots(figsize=(5.7, 3.2), layout='constrained')
    for i in [3, 4]:
        ax.plot(a[f'sigma_{i}'], a[f'q_{i}'], color=PALETTE[i-3],
                ls='-' if i==3 else '--', label=f'{10 if i==3 else 15}°')
    styled(ax, '波数（cm$^{-1}$）', '模型内有效往返系数（1）')
    ax.legend(frameon=False)
    save(fig, 'silicon_q')

    fig, ax = plt.subplots(figsize=(5.7, 3.2), layout='constrained')
    cv = s['blocked_cv']['fold_rmse_pp']
    x=np.arange(1,6)
    ax.bar(x-.17, cv['two'], width=.32, color=PALETTE[1], label='两光束')
    ax.bar(x+.17, cv['multi'], width=.32, color=PALETTE[0], label='多光束')
    styled(ax, '留出折次', '留出均方根误差（百分点）')
    ax.set_xticks(x)
    ax.legend(frameon=False)
    save(fig, 'silicon_cv')


def reliability(sic, si):
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.0), layout='constrained')
    ref = [sic['two_beam']['d_um'], si['main']['parameters'][0]]
    sc = [v for v in sic['sensitivity'] if v['kind']=='index_assumption' and v['model']=='wang']
    sc += [{'scale':1, 'd_um':ref[0]}]
    sc = sorted(sc, key=lambda v:v['scale'])
    for ax, seq, key, d, title in [(axes[0], sc, 'scale', ref[0], '（a）碳化硅'),
                                   (axes[1], si['index_sensitivity'], 'n_scale', ref[1], '（b）硅')]:
        xx = [(v[key]-1)*100 for v in seq]
        yy = [(v['d_um']/d-1)*100 for v in seq]
        ax.plot(xx, yy, 'o-', color=PALETTE[0])
        ax.axhline(0, color='#aaaaaa', lw=.6)
        styled(ax, '折射率相对变化（%）', '厚度相对变化（%）')
        ax.set_title(title)
    save(fig, 'index_sensitivity')

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.0), layout='constrained')
    for j, ax in enumerate(axes):
        if j==0:
            vv=[sic['two_beam']['d_um']]+[v['d_um'] for v in sic['independent_angles']]
        else:
            vv=[si['main']['parameters'][0]]+[v['parameters'][0] for v in si['per_angle']]
        ax.plot(np.arange(3),vv,'o-',color=PALETTE[j])
        for k, v in enumerate(vv):
            offset, align = [((6, -14), 'left'), ((0, 6), 'center'), ((-6, -14), 'right')][k]
            ax.annotate(f'{v:.4f}',(k,v),xytext=offset,textcoords='offset points',ha=align,fontsize=9)
        ax.set_xticks(np.arange(3),['共同','10°','15°'])
        ax.margins(y=.4)
        styled(ax,'拟合方式','厚度（μm）')
        ax.set_title('（a）碳化硅' if j==0 else '（b）硅')
    save(fig,'angle_consistency')

    sb=pd.read_csv(ROOT/'results'/'sic_bootstrap.csv').d_um.to_numpy()
    ib=np.load(ROOT/'results'/'silicon_predictions.npz')['bootstrap_parameters'][:,0]
    fig, axes=plt.subplots(1,2,figsize=(5.9,3.0),layout='constrained')
    for j,(ax,v,d) in enumerate(zip(axes,[sb,ib],ref)):
        ax.hist(v,bins=16,color=PALETTE[j],alpha=.8,edgecolor='white')
        ax.axvline(d,color='#333333',linestyle='--',linewidth=1.1,label='主拟合')
        styled(ax,'条件重抽样厚度（μm）','次数')
        ax.set_title('（a）碳化硅' if j==0 else '（b）硅')
        ax.tick_params(axis='x',labelsize=8)
        ax.legend(frameon=False,fontsize=8)
    save(fig,'bootstrap')


def derived_tables(sic, si):
    metrics=[]
    for material, ids, models in [('SiC',[1,2],['two','airy']),('Si',[3,4],['two','multi'])]:
        ys=[];ps={m:[] for m in models}
        for i in ids:
            if material=='SiC':
                a=pd.read_csv(ROOT/'results'/f'sic_prediction_{i}.csv')
                ys.append(a.reflectance_percent.to_numpy())
                ps['two'].append(a.two_beam_percent.to_numpy())
                ps['airy'].append(a.airy_percent.to_numpy())
            else:
                a=pd.read_csv(ROOT/'results'/f'silicon_prediction_{i}.csv')
                ys.append(a.R_pct.to_numpy())
                ps['two'].append(a.two_pct.to_numpy());ps['multi'].append(a.multi_pct.to_numpy())
        y=np.concatenate(ys)
        sst=sum(np.sum((v-v.mean())**2) for v in ys)
        for model in models:
            r=y-np.concatenate(ps[model])
            metrics.append(dict(material=material,model=model,n=len(y),
                rmse_pp=float(np.sqrt(np.mean(r*r))),mae_pp=float(np.mean(abs(r))),
                max_abs_pp=float(np.max(abs(r))),r2=float(1-np.sum(r*r)/sst)))
    (ROOT/'results'/'derived_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    names={'two':'两光束','airy':'有效多光束核','multi':'多光束'}
    table('metrics','在各自主分析波段上的反射率拟合指标','tab:metrics',
          ['材料与模型','RMSE/百分点','MAE/百分点','$R^2$','最大绝对残差'],
          [[v['material']+names[v['model']],f"{v['rmse_pp']:.5f}",f"{v['mae_pp']:.5f}",
            f"{v['r2']:.6f}",f"{v['max_abs_pp']:.4f}"] for v in metrics])
    table('window_sensitivity','有效波段内的窗口敏感性','tab:windows',
          ['材料',r'窗口/$\mathrm{cm}^{-1}$',r'厚度/$\mu$m','RMSE/百分点'],
          [['SiC',f"{v['window'][0]:.0f}--{v['window'][1]:.0f}",f"{v['d_um']:.5f}",f"{v['rmse_pp']:.5f}"]
           for v in sic['sensitivity'] if v['kind']=='window']+
          [['Si',f"{v['window_cm1'][0]}--{v['window_cm1'][1]}",f"{v['parameters'][0]:.5f}",f"{v['rmse_pp']:.5f}"]
           for v in si['window_sensitivity'][:5]])
    sc=[v for v in sic['sensitivity'] if v['kind']=='index_assumption' and v['model']=='wang']
    table('index_sensitivity','折射率整体缩放的情景比较','tab:index-sensitivity',
          ['材料','折射率缩放',r'厚度/$\mu$m',r'相对主值变化/\%'],
          [['SiC',f"{v['scale']:.3f}",f"{v['d_um']:.5f}",f"{100*(v['d_um']/sic['two_beam']['d_um']-1):+.3f}"] for v in sc]+
          [['Si',f"{v['n_scale']:.3f}",f"{v['d_um']:.5f}",f"{100*(v['d_um']/si['main']['parameters'][0]-1):+.3f}"] for v in si['index_sensitivity']])
    cb=sic['conditional_bootstrap'];ib=si['conditional_bootstrap']
    table('bootstrap','模型固定条件下的移动块残差重抽样','tab:bootstrap',
          ['材料','次数',r'块宽/$\mathrm{cm}^{-1}$',r'厚度标准差/$\mu$m',r'95\%分位范围/$\mu$m'],
          [['SiC',cb['iterations'],'约50',f"{cb['std_d_um']:.6f}",f"[{cb['percentile95_d_um'][0]:.5f}, {cb['percentile95_d_um'][1]:.5f}]"],
           ['Si',ib['replicates'],f"{ib['block_width_cm1']:.1f}",f"{ib['d_std_um']:.6f}",f"[{ib['d_percentiles_2_5_50_97_5_um'][0]:.5f}, {ib['d_percentiles_2_5_50_97_5_um'][2]:.5f}]"]])
    table('cv','五折连续波数块留出验证','tab:cv',
          ['折次','SiC两束','SiC多束核','Si两束','Si多束'],
          [[k+1,f"{sic['cross_validation']['folds'][k]['two_beam_rmse_pp']:.5f}",
            f"{sic['cross_validation']['folds'][k]['airy_rmse_pp']:.5f}",
            f"{si['blocked_cv']['fold_rmse_pp']['two'][k]:.5f}",f"{si['blocked_cv']['fold_rmse_pp']['multi'][k]:.5f}"] for k in range(5)])
    v=sic['reference_index_sample']
    table('sic_index','碳化硅主窗口内的文献折射率及群折射率','tab:sic-index',
          [r'波数/$\mathrm{cm}^{-1}$','$n$',r'$n+\sigma\,\mathrm{d}n/\mathrm{d}\sigma$'],
          [[f'{x:.0f}',f'{n:.6f}',f'{ng:.6f}'] for x,n,ng in zip(v['wavenumber'],v['n'],v['group_index'])])
    table('sic_peaks','碳化硅峰位与恢复的相对条纹级次','tab:sic-peaks',
          ['相对级次',r'附件1峰位/$\mathrm{cm}^{-1}$',r'附件2峰位/$\mathrm{cm}^{-1}$'],
          [[order,f'{a:.3f}',f'{b:.3f}'] for order,a,b in zip(sic['initializers'][0]['fringe_orders'],
              sic['initializers'][0]['maxima_cm_inverse'],sic['initializers'][1]['maxima_cm_inverse'])])
    pp=si['main']['parameters'];cal=si['main']['calibration_gain_offset']
    table('silicon_parameters','硅共同反演的参数记录','tab:si-parameters',
          ['参数','估计值','单位'],
          [[r'$d$',f'{pp[0]:.6f}',r'$\mu$m'],
           [r'$\Omega_p$',f'{pp[1]:.4f}',r'$\mathrm{cm}^{-1}$'],
           [r'$\Gamma$',f'{pp[2]:.4f}',r'$\mathrm{cm}^{-1}$'],
           [r'$g$',f'{pp[3]:.6f}','1'],
           [r'$a_{10}$',f'{cal[0][1]:.6f}','百分点'],
           [r'$b_{10}$',f'{cal[0][0]:.6f}','1'],
           [r'$a_{15}$',f'{cal[1][1]:.6f}','百分点'],
           [r'$b_{15}$',f'{cal[1][0]:.6f}','1']],spec='lrl')
    rt=si['model_based_roundtrip'];left=rt[:5];right=rt[5:]
    table('silicon_roundtrip','硅模型内的分波段平均往返保持量级','tab:si-roundtrip',
          [r'窗口/$\mathrm{cm}^{-1}$',r'$10^\circ$中位数',r'$15^\circ$中位数'],
          [[f"{x['window_cm1'][0]}--{x['window_cm1'][1]}",f"{x['q_median']:.5f}",f"{y['q_median']:.5f}"]
           for x,y in zip(left,right)])
    c=si['numerical_self_checks']
    def scientific(value):
        if value == 0:
            return '0'
        mantissa, exponent = f'{value:.2e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    table('physical_checks','用于代码核对的物理极限与无噪声算例','tab:physical-checks',
          ['检查内容','误差量'],
          [['闭式振幅与80束显式叠加',scientific(c['explicit_80_ray_field_max_error'])],
           ['零界面对比度时的厚度响应',scientific(c['zero_contrast_thickness_invariance_error_pp'])],
           ['无噪声参数恢复最大相对误差',scientific(c['synthetic_max_relative_parameter_error'])]],spec='lr')
    table('answer_summary','主结果及双角独立估计汇总','tab:answer-summary',
          ['材料',r'共同厚度/$\mu$m',r'$10^\circ$独立/$\mu$m',r'$15^\circ$独立/$\mu$m','主模型'],
          [['SiC',f"{sic['two_beam']['d_um']:.5f}",f"{sic['independent_angles'][0]['d_um']:.5f}",f"{sic['independent_angles'][1]['d_um']:.5f}",'色散两光束'],
           ['Si',f"{si['main']['parameters'][0]:.5f}",f"{si['per_angle'][0]['parameters'][0]:.5f}",f"{si['per_angle'][1]['parameters'][0]:.5f}",'有效多光束']],spec='lrrrl')


def main():
    configure()
    sic=json.loads((ROOT/'results'/'sic_results.json').read_text(encoding='utf-8'))
    si=json.loads((ROOT/'results'/'silicon_results.json').read_text(encoding='utf-8'))
    sic_figures(sic);silicon_figures(si);reliability(sic,si);derived_tables(sic,si)


if __name__=='__main__':
    main()

"""L9 three-factor scenario sensitivity for the two unchanged main models.

Run after both main analyses: python code/joint_sensitivity.py
Writes results/joint_sensitivity.json and tables/joint_sensitivity.tex only.
No bootstrap, PDF, or Word output is produced. Input scale/angle ranges are
user-defined scenarios, not probabilistic uncertainty distributions.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sic_analysis as sic
import silicon_analysis as si

ROOT = Path(__file__).resolve().parents[1]
SCALES = (.99, 1., 1.01)
SHIFTS = (-.2, 0., .2)
SIC_WINDOWS = ((2000., 3100.), (2000., 3300.), (2100., 3300.))
SI_WINDOWS = ((450., 3500.), (450., 4000.), (600., 4000.))
# Standard strength-two OA(9,3^3): every pair of factor levels occurs once.
L9 = np.array([[1,1,1], [1,2,2], [1,3,3],
               [2,1,2], [2,2,3], [2,3,1],
               [3,1,3], [3,2,1], [3,3,2]], dtype=int)


def read_sic():
    result = []
    for i in (1,2):
        a = pd.read_excel(ROOT / 'data' / f'附件{i}.xlsx').to_numpy(float)
        a = a[np.isfinite(a).all(axis=1) & (a[:, 1] != 0)]
        if not np.all(np.diff(a[:, 0]) > 0):
            raise ValueError('SiC wavenumbers must increase without duplicates.')
        result.append(a)
    return result


def range_analysis(rows, column, baseline):
    analysis = []
    for factor, name in enumerate(['index_scale', 'common_angle_shift_deg', 'window_level']):
        means = [float(np.mean([r[column] for r in rows if r['levels'][factor] == lev]))
                 for lev in (1,2,3)]
        span = max(means) - min(means)
        analysis.append(dict(factor=name, level_means_d_um=means,
                             level_mean_range_um=span,
                             level_mean_range_relative_percent=100*span/baseline))
    return analysis


def write_table(rows, ranges, destination):
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{三因素三水平 L9 联合情景敏感性；变化均相对于各材料主模型}',
             r'\label{tab:joint-sensitivity}', r'\small',
             r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{crrrrrrr}', r'\toprule',
             r'情景 & $s_n$ & $\Delta\theta/{}^\circ$ & 窗口级别 & $d_{\rm SiC}/\mu\mathrm m$ & 变化/\% & $d_{\rm Si}/\mu\mathrm m$ & 变化/\%\\',
             r'\midrule']
    for row in rows:
        lines.append(f"{row['scenario']} & {row['index_scale']:.2f} & {row['angle_shift_deg']:+.1f} & "
                     f"{row['levels'][2]} & {row['sic_d_um']:.5f} & {row['sic_relative_change_percent']:+.3f} & "
                     f"{row['silicon_d_um']:.5f} & {row['silicon_relative_change_percent']:+.3f} \\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\begin{tabular}{lrrrr}', r'\toprule',
              r'因素 & SiC 极差/$\mu\mathrm m$ & SiC 极差/\% & Si 极差/$\mu\mathrm m$ & Si 极差/\%\\',
              r'\midrule']
    for i, label in enumerate(['折射率比例', '共同角度偏移', '波段选择']):
        a, b = ranges['sic'][i], ranges['silicon'][i]
        lines.append(f"{label} & {a['level_mean_range_um']:.6f} & {a['level_mean_range_relative_percent']:.4f} & "
                     f"{b['level_mean_range_um']:.6f} & {b['level_mean_range_relative_percent']:.4f} \\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\begin{minipage}{0.98\textwidth}\footnotesize',
              r'窗口级别1、2、3对SiC分别表示$2000$--$3100$、$2000$--$3300$、$2100$--$3300$；'
              r'对Si分别表示$450$--$3500$、$450$--$4000$、$600$--$4000$，单位均为$\mathrm{cm}^{-1}$。'
              r'SiC各情景均剔除$2300$--$2400$。$s_n$同时乘入同一参照折射率函数；'
              r'两个角度均增加$\Delta\theta$。SiC固定两束模型，Si固定含Drude衬底及有效往返包络的多束模型，'
              r'每个情景重新估计原模型中的其余参数。',
              '因素极差为该因素三个水平的平均厚度之最大值减最小值；百分比以各材料主厚度归一化。'
              '这些人为指定的范围不代表概率分布或置信区间。L9中交互效应可能与主效应混杂，'
              '该表不能独立识别交互作用、因果贡献或全局Sobol指数。',
              r'\end{minipage}', r'\end{table}', '']
    # Rows require two literal backslashes in TeX; f-string escaping above
    # creates one, so normalize only the generated numerical data rows.
    lines = [line + '\\' if line.endswith(' \\') and not line.endswith(' \\\\') else line
             for line in lines]
    destination.write_text('\n'.join(lines), encoding='utf-8')


def main():
    start = time.perf_counter()
    r_sic = json.loads((ROOT/'results'/'sic_results.json').read_text(encoding='utf-8'))
    r_si = json.loads((ROOT/'results'/'silicon_results.json').read_text(encoding='utf-8'))
    init_sic = r_sic['two_beam']['parameters']
    init_si = r_si['main']['parameters']
    baseline_sic, baseline_si = float(init_sic[0]), float(init_si[0])
    raw_sic = read_sic()
    raw_si, _ = si.load_inputs(ROOT/'data')
    # Check level balance and pairwise coverage instead of assuming the table.
    for i in range(3):
        assert np.array_equal(np.bincount(L9[:, i])[1:], [3,3,3])
        for j in range(i+1,3):
            assert len({tuple(r) for r in L9[:, [i,j]]}) == 9
    fs0 = sic.fit(sic.select(raw_sic, SIC_WINDOWS[1]), angles=sic.ANGLES,
                  window=SIC_WINDOWS[1], model='wang', scale=1., degree=3,
                  envelope_degree=2, airy=False, initial=init_sic, multistart=False)
    fi0, _, _ = si.fit_model(si.select(raw_si, *SI_WINDOWS[1]), angles=si.ANGLES,
                            kind='multi', initial=init_si, n_scale=1., multistart=False)
    checks = dict(sic_d_difference_um=fs0['d_um']-baseline_sic,
                  silicon_d_difference_um=fi0['parameters'][0]-baseline_si,
                  sic_rmse_difference_pp=fs0['rmse_pp']-r_sic['two_beam']['rmse_pp'],
                  silicon_rmse_difference_pp=fi0['rmse_pp']-r_si['main']['rmse_pp'])
    if any(abs(checks[k]) > 1e-5 for k in ('sic_d_difference_um', 'silicon_d_difference_um')):
        raise RuntimeError(f'Reference refit does not reproduce main thickness: {checks}')
    if any(abs(checks[k]) > 1e-7 for k in ('sic_rmse_difference_pp', 'silicon_rmse_difference_pp')):
        raise RuntimeError(f'Reference refit does not reproduce main RMSE: {checks}')
    print('Baseline equivalence:', checks, flush=True)
    rows = []
    for num, levels in enumerate(L9, start=1):
        a,b,c = levels - 1
        scale, shift = SCALES[a], SHIFTS[b]
        sw, iw = SIC_WINDOWS[c], SI_WINDOWS[c]
        angles = np.asarray(sic.ANGLES) + shift
        tick = time.perf_counter()
        fs = sic.fit(sic.select(raw_sic, sw), angles=angles, window=sw,
                     model='wang', scale=scale, degree=3, envelope_degree=2,
                     airy=False, initial=init_sic, multistart=False)
        fi, _, _ = si.fit_model(si.select(raw_si, *iw), angles=angles,
                               kind='multi', initial=init_si, n_scale=scale,
                               multistart=False)
        if not (fs['success'] and fi['success']):
            raise RuntimeError(f'Scenario {num} did not converge.')
        if any(fi['active_bounds']):
            raise RuntimeError(f'Silicon scenario {num} reached a parameter bound.')
        ds, di = fs['d_um'], fi['parameters'][0]
        row = dict(scenario=num, levels=levels.tolist(), index_scale=scale,
                   angle_shift_deg=shift, angles_deg=angles.tolist(),
                   sic_window_cm_inverse=list(sw), silicon_window_cm_inverse=list(iw),
                   sic_d_um=ds, silicon_d_um=di,
                   sic_relative_change_percent=100*(ds/baseline_sic-1),
                   silicon_relative_change_percent=100*(di/baseline_si-1),
                   sic_rmse_pp=fs['rmse_pp'], silicon_rmse_pp=fi['rmse_pp'],
                   sic_n_points=fs['n'], silicon_n_points=fi['n_points'],
                   sic_parameters=fs['parameters'].tolist(), silicon_parameters=fi['parameters'],
                   sic_linear_coefficients=[v.tolist() for v in fs['coefficients']],
                   silicon_calibration_gain_offset=fi['calibration_gain_offset'],
                   elapsed_seconds=time.perf_counter()-tick)
        rows.append(row)
        print(f"L9 {num}: SiC {ds:.8f} um, Si {di:.8f} um ({row['elapsed_seconds']:.2f} s)",flush=True)
    ranges = dict(sic=range_analysis(rows, 'sic_d_um', baseline_sic),
                  silicon=range_analysis(rows, 'silicon_d_um', baseline_si))
    result = dict(design='L9(3^3) orthogonal scenario array, strength two',
                  reference_thickness_um=dict(sic=baseline_sic, silicon=baseline_si),
                  levels=dict(index_scale=SCALES, common_angle_shift_deg=SHIFTS,
                              sic_windows_cm_inverse=SIC_WINDOWS, silicon_windows_cm_inverse=SI_WINDOWS),
                  sic_excluded_local_band_cm_inverse=list(sic.EXCLUDED),
                  fixed_models=dict(sic='Wang 4H ordinary ray, two beam, Chebyshev degree 3, amplitude degree 2',
                                    silicon='Chandler-Horowitz film, Drude effective substrate, Airy multiple reflections, free effective roundtrip envelope'),
                  baseline_equivalence_checks=checks, scenarios=rows, factor_ranges=ranges,
                  observed_scenario_ranges_um=dict(sic=[min(r['sic_d_um'] for r in rows),max(r['sic_d_um'] for r in rows)],
                                                   silicon=[min(r['silicon_d_um'] for r in rows),max(r['silicon_d_um'] for r in rows)]),
                  limitations=[
                      'Factor ranges were chosen by the analyst and are not probability distributions or confidence bounds.',
                      'Factors use three level means; ranges describe these scenarios, not independent causal contributions.',
                      'L9 covers nine of the 27 Cartesian combinations; interactions cannot be independently identified or separated from main effects.',
                      'This is not a global Sobol analysis, a probabilistic metrology budget, or a bootstrap experiment.',
                      'Nuisance parameters are refitted with the exact original model; all nonlinear starts use the stored main fit.',
                      'Refractive-index scaling modifies the fixed reference dispersion but does not replace it with a different material law.' ],
                  elapsed_seconds=time.perf_counter()-start)
    (ROOT/'results'/'joint_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_table(rows,ranges,ROOT/'tables'/'joint_sensitivity.tex')
    print('Total elapsed seconds:',result['elapsed_seconds'],flush=True)


if __name__ == '__main__':
    main()

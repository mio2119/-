"""Reproducible joint Si-film inversion for attachments 3 and 4.

Only reads inputs and writes JSON/NPZ/CSV, never PDF or DOCX.  Dependencies:
numpy, pandas, scipy, openpyxl (only if CSV input is absent).
Run: python code/silicon_analysis.py --bootstrap 120

The film index is fixed to Chandler-Horowitz & Amirtharaj (2005),
doi:10.1063/1.1923612.  The substrate Drude response and the round-trip
envelope are effective nuisance models, not independent measurements of
carrier density, absorption, roughness, or coherence length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# A local dependency directory is optional and is never created by this script.
if (ROOT / 'scratch' / 'sic_deps').is_dir():
    sys.path.insert(0, str(ROOT / 'scratch' / 'sic_deps'))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import wilcoxon

ANGLES = (10.0, 15.0)
BOUNDS = (np.array([1.5, 100., 1., 0.]), np.array([6., 15000., 5000., 2.]))
DEFAULT_P = np.array([3.4, 3900., 370., .10])


def silicon_eps(sigma: np.ndarray, n_scale: float = 1.) -> np.ndarray:
    """Room-temperature intrinsic Si index squared; lambda is in micrometres."""
    wavelength = 1.e4 / sigma
    return n_scale**2 * (11.67316 + 1. / wavelength**2
                         + .004482633 / (wavelength**2 - 1.108205**2))


def field_terms(sigma, angle, p, n_scale=1.):
    d, wp, gamma, eta = p
    ef = silicon_eps(sigma, n_scale)
    es = ef - wp**2 / (sigma * (sigma + 1j * gamma))
    sn, cs = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    uf, us = np.sqrt(ef - sn**2 + 0j), np.sqrt(es - sn**2 + 0j)
    propagation = np.exp(4j * np.pi * d * 1.e-4 * sigma * uf
                         - eta * (sigma / 1000.)**2)
    rs01, rs12 = (cs - uf) / (cs + uf), (uf - us) / (uf + us)
    rp01 = (ef * cs - uf) / (ef * cs + uf)
    rp12 = (es * uf - ef * us) / (es * uf + ef * us)
    return propagation, ((rs01, rs12), (rp01, rp12))


def optical_reflectance(sigma, angle, p, kind='multi', n_scale=1.):
    """Return unpolarized reflectance in percentage points before calibration.

    kind='two' retains exactly the surface ray and the first substrate ray.
    The product t01*t10=1-r01**2 is retained, so this is a field truncation
    of the same model rather than a separately parameterized cosine curve.
    """
    E, terms = field_terms(sigma, angle, p, n_scale)
    reflected = []
    for r01, r12 in terms:
        if kind == 'multi':
            r = (r01 + r12 * E) / (1. + r01 * r12 * E)
        elif kind == 'two':
            r = r01 + (1. - r01**2) * r12 * E
        else:
            raise ValueError(kind)
        reflected.append(np.abs(r)**2)
    return 50. * (reflected[0] + reflected[1])


def calibration(pred, y):
    centered = pred - np.mean(pred)
    gain = np.dot(centered, y - np.mean(y)) / np.dot(centered, centered)
    offset = np.mean(y) - gain * np.mean(pred)
    return np.array([gain, offset])


def fit_model(zs, angles=ANGLES, kind='multi', initial=None, n_scale=1.,
              multistart=False, eta_fixed=None):
    """Profile out per-angle linear gain and offset before nonlinear fitting."""
    initial = DEFAULT_P.copy() if initial is None else np.array(initial).copy()
    free = np.array([0, 1, 2, 3]) if eta_fixed is None else np.array([0, 1, 2])
    def unpack(v):
        p = initial.copy()
        p[free] = v
        if eta_fixed is not None:
            p[3] = eta_fixed
        return p
    def residual(v, details=False):
        p = unpack(v)
        preds, calibrations = [], []
        for z, angle in zip(zs, angles):
            pred = optical_reflectance(z[:, 0], angle, p, kind, n_scale)
            a = calibration(pred, z[:, 1])
            calibrations.append(a)
            preds.append(a[0] * pred + a[1])
        res = np.concatenate([pred - z[:, 1] for pred, z in zip(preds, zs)])
        return (res, preds, calibrations) if details else res
    starts = [initial]
    if multistart:
        starts += [np.array([d, wp, 350., .08]) for d in [2.8, 3.2, 3.6, 4.0, 4.4]
                   for wp in [3000., 4500.]]
    fits = []
    for p0 in starts:
        f = least_squares(residual, p0[free], bounds=(BOUNDS[0][free], BOUNDS[1][free]),
                          x_scale='jac', max_nfev=700, ftol=1e-10,
                          xtol=1e-10, gtol=1e-10)
        fits.append(f)
    fit = min(fits, key=lambda f: f.cost)
    res, preds, calib = residual(fit.x, True)
    p = unpack(fit.x)
    npar = len(free) + 2 * len(zs)
    sse = float(np.dot(res, res))
    rmse = float(np.sqrt(np.mean(res**2)))
    jac = fit.jac
    norms = np.sqrt(np.sum(jac**2, axis=0))
    singular = np.linalg.svd(jac / norms, compute_uv=False)
    cov = np.linalg.pinv(jac.T @ jac) * sse / (len(res) - npar)
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    result = dict(parameters=p.tolist(), parameter_names=['d_um', 'wp_cm-1', 'gamma_cm-1', 'eta'],
                  n_points=len(res), n_parameters=npar, rmse_pp=rmse, sse_pp2=sse,
                  bic_iid_descriptive=float(len(res) * np.log(sse / len(res))
                                            + npar * np.log(len(res))),
                  calibration_gain_offset=[a.tolist() for a in calib],
                  angle_rmse_pp=[float(np.sqrt(np.mean((pred-z[:, 1])**2)))
                                 for pred, z in zip(preds, zs)],
                  scaled_jacobian_condition=float(singular[0] / singular[-1]),
                  profiled_parameter_correlation=corr.tolist(),
                  naive_iid_parameter_se=se.tolist(), success=bool(fit.success),
                  message=str(fit.message),
                  active_bounds=np.asarray(fit.active_mask).tolist(),
                  starts_final_d_cost=[[float(unpack(f.x)[0]),float(f.cost)] for f in fits])
    return result, preds, calib


def select(zs, low, high, stride=1):
    return [z[(z[:, 0] >= low) & (z[:, 0] <= high)][::stride] for z in zs]


def predicted(zs, angles, p, calib, kind='multi', n_scale=1.):
    return [a[0] * optical_reflectance(z[:, 0], angle, p, kind, n_scale) + a[1]
            for z, angle, a in zip(zs, angles, calib)]


def cross_validation(zs, initial, block_width=100., k=5):
    """Hold out 100 cm^-1 contiguous blocks; both angles use the same folds."""
    errors = {kind: [] for kind in ['two', 'multi']}
    block_loss = {kind: {} for kind in ['two', 'multi']}
    masks = [np.floor((z[:, 0] - 450.) / block_width).astype(int) for z in zs]
    for fold in range(k):
        train = [z[b % k != fold] for z, b in zip(zs, masks)]
        test = [z[b % k == fold] for z, b in zip(zs, masks)]
        test_blocks = [b[b % k == fold] for b in masks]
        for kind in errors:
            out, _, a = fit_model(train, kind=kind, initial=initial)
            yh = predicted(test, ANGLES, out['parameters'], a, kind)
            rs = [pred-z[:, 1] for pred, z in zip(yh, test)]
            errors[kind].append(float(np.sqrt(np.mean(np.concatenate(rs)**2))))
            for b in np.unique(np.concatenate(test_blocks)):
                vals = np.concatenate([r[tb == b] for r, tb in zip(rs, test_blocks)])
                block_loss[kind][int(b)] = float(np.mean(vals**2))
    keys = sorted(block_loss['two'])
    diff = np.array([block_loss['two'][b] - block_loss['multi'][b] for b in keys])
    test = wilcoxon(diff, alternative='greater', method='auto')
    return dict(block_width_cm1=block_width, folds=k, fold_rmse_pp=errors,
                block_ids=keys, block_mse_pp2=block_loss,
                multi_better_blocks=int(np.sum(diff > 0)), n_blocks=len(diff),
                median_block_mse_reduction_pp2=float(np.median(diff)),
                paired_wilcoxon_p_approx=float(test.pvalue),
                warning='Adjacent blocks may still be correlated; p is supplementary, not a physical proof.')


def bootstrap(zs, pred, initial, replicates, block_length=100, seed=20250905):
    """Paired circular moving-block residual bootstrap, conditional on model.

    Uses the same sampled blocks for both angles to preserve their residual
    dependence. This quantifies fit variability only, not model discrepancy,
    refractive-index uncertainty, or instrument uncertainty.
    """
    rng = np.random.default_rng(seed)
    n = len(zs[0])
    if any(len(z) != n for z in zs):
        raise ValueError('Bootstrap assumes aligned angle grids.')
    residuals = [z[:, 1]-y for z, y in zip(zs, pred)]
    residuals = [r-r.mean() for r in residuals]
    pars = []
    for j in range(replicates):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
        ix = np.concatenate([(s+np.arange(block_length)) % n for s in starts])[:n]
        sample = [np.column_stack([z[:, 0], y+r[ix]]) for z, y, r in zip(zs,pred,residuals)]
        out, _, _ = fit_model(sample, initial=initial)
        pars.append(out['parameters'])
        if (j+1) % 30 == 0:
            print(f'Silicon bootstrap: {j+1}/{replicates}', flush=True)
    arr = np.asarray(pars)
    return dict(replicates=replicates, seed=seed, block_length_points=block_length,
                block_width_cm1=float(np.median(np.diff(zs[0][:,0]))*block_length),
                d_percentiles_2_5_50_97_5_um=np.percentile(arr[:,0],[2.5,50,97.5]).tolist(),
                d_std_um=float(np.std(arr[:,0],ddof=1)),
                warning='Conditional resampling interval, not absolute metrology uncertainty.'), arr


def q_diagnostics(sigma, p):
    bands = [(450,800),(800,1200),(1200,2000),(2000,3000),(3000,4000)]
    out = []
    for angle in ANGLES:
        E, terms = field_terms(sigma, angle, p)
        qpol = [np.abs(r01*r12*E) for r01,r12 in terms]
        for low, high in bands:
            ix = (sigma >= low)&(sigma <= high)
            q = np.mean(qpol,axis=0)[ix]
            out.append(dict(angle_deg=angle, window_cm1=[low,high],
                            q_median=float(np.median(q)),q_max=float(np.max(q)),
                            omitted_field_scale_indicator_median=float(np.median(q/(1-q)))))
    return out


def harmonic_diagnostics(zs, p, pred):
    """Describe phase-locked higher-harmonic evidence after a smooth baseline.

    Each of two windows is fitted by the same cubic baseline and quadratic
    envelopes, with K=1 versus K=2. Phase comes from the fitted optical model.
    This is a secondary nonnested descriptive check, not the thickness solver.
    """
    results = []
    for lo,hi in [(550.,1500.),(1500.,2800.)]:
        for z,angle in zip(zs,ANGLES):
            zz=z[(z[:,0]>=lo)&(z[:,0]<=hi)]
            x,y=zz.T;t=(2*x-lo-hi)/(hi-lo)
            E, terms=field_terms(x,angle,p)
            # At low angle s and p phases differ by a constant pi, absorbed
            # by the free sine/cosine coefficients.
            phi=np.unwrap(np.angle(terms[0][1]*E))
            models=[]
            for K in [1,2]:
                A=np.column_stack([t**j for j in range(4)]+
                    [t**j*c for h in range(1,K+1)
                     for c in [np.cos(h*phi),np.sin(h*phi)] for j in range(3)])
                coeff=np.linalg.lstsq(A,y,rcond=None)[0]
                rmse=float(np.sqrt(np.mean((A@coeff-y)**2)))
                amps=[float(np.hypot(coeff[4+6*h],coeff[7+6*h])) for h in range(K)]
                models.append(dict(K=K,rmse_pp=rmse,center_amplitudes_pp=amps))
            results.append(dict(angle_deg=angle,window_cm1=[lo,hi],models=models,
                second_to_first_amplitude_ratio=models[1]['center_amplitudes_pp'][1]/models[1]['center_amplitudes_pp'][0]))
    return results


def load_inputs(data_dir):
    zs, audit = [], []
    for attachment in [3,4]:
        csv = data_dir / f'spectrum_{attachment}.csv'
        file = csv if csv.exists() else data_dir/f'附件{attachment}.xlsx'
        data = pd.read_csv(file) if file.suffix == '.csv' else pd.read_excel(file)
        z = data.iloc[:,:2].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        bad = np.any(~np.isfinite(z),axis=1)
        finite = z[~bad]
        duplicate_count = int(pd.Series(finite[:,0]).duplicated().sum())
        if duplicate_count:
            raise ValueError(f'Duplicate wavenumbers in {file.name}')
        finite=finite[np.argsort(finite[:,0])]
        audit.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            raw_rows=len(z),nonfinite_rows=int(np.sum(bad)),duplicates=duplicate_count,
            min_sigma_cm1=float(finite[0,0]),max_sigma_cm1=float(finite[-1,0]),
            spacing_median_cm1=float(np.median(np.diff(finite[:,0]))),
            zero_reflectance_rows=np.flatnonzero(finite[:,1]==0).tolist(),
            fit_window_cm1=[450,4000],fit_rows=int(np.sum((finite[:,0]>=450)&(finite[:,0]<=4000))),
            endpoint_note='Both files start with R=0 at 399.6747 cm^-1; excluded with the out-of-literature-range low end.'))
        zs.append(finite)
    return zs,audit


def numerical_self_checks():
    """Independent field-series, zero-contrast and inverse recovery checks."""
    x=np.linspace(500.,3900.,401)
    p=np.array([3.42,3850.,380.,.10])
    E,terms=field_terms(x,10.,p)
    errors=[]
    for r01,r12 in terms:
        exact=(r01+r12*E)/(1+r01*r12*E)
        term=(1-r01**2)*r12*E
        series=r01.copy()
        for _ in range(80):
            series=series+term
            term=term*(-r01*r12*E)
        errors.append(float(np.max(abs(exact-series))))
    zero_a=optical_reflectance(x,10.,[2.,0.,380.,0.])
    zero_b=optical_reflectance(x,10.,[5.,0.,380.,0.])
    synthetic=[np.column_stack([x,a*optical_reflectance(x,th,p)+b])
               for th,a,b in [(10.,.94,-.2),(15.,1.04,-.5)]]
    fit,_,_=fit_model(synthetic,initial=[3.3,3500.,320.,.08])
    relative_error=np.max(abs((np.array(fit['parameters'])-p)/p))
    assert max(errors)<1.e-12, 'Airy / explicit-ray-sum check failed'
    assert np.max(abs(zero_a-zero_b))<1.e-12, 'Zero-contrast limit failed'
    assert relative_error<1.e-7, 'Noise-free synthetic inverse recovery failed'
    return dict(explicit_80_ray_field_max_error=max(errors),
                zero_contrast_thickness_invariance_error_pp=float(np.max(abs(zero_a-zero_b))),
                synthetic_max_relative_parameter_error=float(relative_error),
                synthetic_true_parameters=p.tolist(),synthetic_recovered_parameters=fit['parameters'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--bootstrap',type=int,default=120)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw,audit=load_inputs(args.data_dir)
    zs=select(raw,450,4000)
    main_fit,yp,calib=fit_model(zs,multistart=True)
    two_fit,yp2,calib2=fit_model(zs,kind='two',multistart=True)
    no_envelope,yp0,c0=fit_model(zs,initial=main_fit['parameters'],eta_fixed=0.)
    print('Joint Si thickness:',main_fit['parameters'][0],'um; RMSE:',main_fit['rmse_pp'],flush=True)
    print('Two-ray RMSE:',two_fit['rmse_pp'],flush=True)
    angle_fits=[]
    for z,angle in zip(zs,ANGLES):
        f,yh,a=fit_model([z],angles=[angle],initial=main_fit['parameters'])
        angle_fits.append(dict(angle_deg=angle,**f))
    windows=[]
    for low,high in [(500,3900),(600,3900),(800,3900),(1000,3900),(1500,3900),(600,3500)]:
        f,_,_=fit_model(select(raw,low,high),initial=main_fit['parameters'])
        windows.append(dict(window_cm1=[low,high],**f))
    index_sensitivity=[]
    for scale in [.99,.995,1.,1.005,1.01]:
        f,_,_=fit_model(zs,initial=main_fit['parameters'],n_scale=scale)
        index_sensitivity.append(dict(n_scale=scale,d_um=f['parameters'][0],rmse_pp=f['rmse_pp']))
    angle_sensitivity=[]
    for delta in [-.2,0.,.2]:
        f,_,_=fit_model(zs,initial=main_fit['parameters'],angles=np.array(ANGLES)+delta)
        angle_sensitivity.append(dict(common_angle_shift_deg=delta,d_um=f['parameters'][0],rmse_pp=f['rmse_pp']))
    cv=cross_validation(zs,main_fit['parameters'])
    # Bootstrap every third point: blocks still span about 145 cm^-1.
    bz=[z[::3] for z in zs];bp=[y[::3] for y in yp]
    boot,bootparams=bootstrap(bz,bp,main_fit['parameters'],args.bootstrap)
    sensitivity_d=[w['parameters'][0] for w in windows]+[f['parameters'][0] for f in angle_fits]
    result=dict(material='Si',thickness_unit='micrometre',reflectance_unit='percentage point',
        numerical_self_checks=numerical_self_checks(),
        data_audit=audit,main=main_fit,two_ray=two_fit,without_envelope=no_envelope,
        per_angle=angle_fits,window_sensitivity=windows,index_sensitivity=index_sensitivity,
        angle_sensitivity=angle_sensitivity,blocked_cv=cv,conditional_bootstrap=boot,
        model_based_roundtrip=q_diagnostics(zs[0][:,0],main_fit['parameters']),
        harmonic_diagnostics=harmonic_diagnostics(zs,main_fit['parameters'],yp),
        reporting=dict(recommended_rounded_d_um=round(main_fit['parameters'][0],2),
            between_angle_absolute_difference_um=abs(angle_fits[0]['parameters'][0]-angle_fits[1]['parameters'][0]),
            window_and_angle_range_um=[min(sensitivity_d),max(sensitivity_d)],
            conclusion='Strong support for a multiple-reflection effective model; coherence conditions cannot be independently verified from these files.',
            limitations=['Literature intrinsic Si n is fixed; actual doping-dependent film n is not independently measured.',
                'Drude substrate parameters and eta are effective nuisance fits, not measured carrier density or coherence length.',
                'Affine per-angle calibration absorbs unknown intensity scale and offset.',
                'Angle-specific fits disagree beyond conditional numerical precision; report this rather than pooling into a falsely narrow interval.',
                'Spectral resolution, beam divergence, spot alignment, temperature, and repeat measurements are not supplied.',
                'The exp(-eta*(sigma/1000)^2) factor models effective per-roundtrip amplitude attenuation; it is not claimed to be a unique physical mechanism.',
                'Naive iid standard errors/BIC and harmonic ratios are diagnostic only; blocked validation and sensitivity are emphasized.']),
        equations=dict(film_index='n_f^2=11.67316+lambda^-2+0.004482633/(lambda^2-1.108205^2), lambda=10000/sigma micrometre',
            substrate='epsilon_s=epsilon_f-wp^2/[sigma*(sigma+i*gamma)]',
            propagation='E=exp(4*pi*i*d*1e-4*sigma*sqrt(epsilon_f-sin(theta)^2))*exp(-eta*(sigma/1000)^2)',
            multi='r=(r01+r12*E)/(1+r01*r12*E)',
            two='r=r01+(1-r01^2)*r12*E',measurement='R_j=a_j+b_j*50*(|r_s|^2+|r_p|^2)'),
        references=[dict(authors='D. Chandler-Horowitz and P. M. Amirtharaj',year=2005,
            title='High-accuracy, midinfrared (450 cm^-1 <= w <= 4000 cm^-1) refractive index values of silicon',
            journal='Journal of Applied Physics',volume='97',article='123526',doi='10.1063/1.1923612',
            original_author_url='https://www.nist.gov/publications/high-accuracy-midinfrared450-cm-1-w-4000-cm-1refractive-index-values-silicon',
            formula_transcription_url='https://refractiveindex.info/?shelf=main&book=Si&page=Chandler-Horowitz')])
    output=args.output_dir/'silicon_results.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays=dict(bootstrap_parameters=bootparams,fit_parameters=np.asarray(main_fit['parameters']))
    for i,(rawz,z,m,t,cal) in enumerate(zip(raw,zs,yp,yp2,calib),start=3):
        arrays[f'raw_sigma_{i}']=rawz[:,0];arrays[f'raw_R_{i}']=rawz[:,1]
        arrays[f'sigma_{i}']=z[:,0];arrays[f'R_{i}']=z[:,1]
        arrays[f'multi_{i}']=m;arrays[f'two_{i}']=t
        arrays[f'residual_multi_{i}']=z[:,1]-m;arrays[f'residual_two_{i}']=z[:,1]-t
        fi=angle_fits[i-3]
        single=predicted([z],[ANGLES[i-3]],fi['parameters'],fi['calibration_gain_offset'])[0]
        arrays[f'separate_angle_{i}']=single
        E,terms=field_terms(z[:,0],ANGLES[i-3],main_fit['parameters'])
        arrays[f'q_{i}']=np.mean([abs(a*b*E) for a,b in terms],axis=0)
        smooth=savgol_filter(rawz[:,1],31,3)
        peaks=find_peaks(smooth,prominence=.1,distance=80)[0]
        arrays[f'peak_sigma_{i}']=rawz[peaks,0]
        pd.DataFrame({'sigma_cm1':z[:,0],'R_pct':z[:,1],'multi_pct':m,'two_pct':t,
                      'separate_angle_pct':single,'residual_multi_pp':z[:,1]-m})\
            .to_csv(args.output_dir/f'silicon_prediction_{i}.csv',index=False)
    np.savez_compressed(args.output_dir/'silicon_predictions.npz',**arrays)
    print('Saved',output,flush=True)
    print('Per angle:',[f['parameters'][0] for f in angle_fits],flush=True)
    print('Conditional bootstrap:',boot,flush=True)


if __name__=='__main__':
    main()

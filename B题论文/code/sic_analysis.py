"""Reproducible phase-based thickness inversion for attachments 1 and 2 (SiC).

Run from any directory: python code/sic_analysis.py [--bootstrap 120]
Dependencies: numpy, pandas, openpyxl, scipy. No PDF or Word is produced.

The reference geometric thickness is conditional on the Wang et al. 4H-SiC
ordinary-ray Sellmeier law. Crystal polytype, polarization, free-carrier density,
and calibration are not given in the supplied measurements. This calculation
therefore separates conditional residual uncertainty from model sensitivity.

Reflectance remains in PERCENTAGE POINTS, never fractions. Wavenumber is cm^-1;
thickness is micrometres and is multiplied by 1e-4 in the phase expression.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parents[1]
ANGLES = (10.0, 15.0)
MAIN_WINDOW = (2000.0, 3300.0)
EXCLUDED = (2300.0, 2400.0)
SEED = 20250907


def index(v, model="wang", scale=1.0):
    """Real-index assumptions; only evaluated away from the phonon band."""
    v = np.asarray(v, dtype=float)
    if model == "wang":
        lam2 = (1e4 / v) ** 2
        eps = (1 + .20075 * lam2 / (lam2 + 12.07224)
               + 5.54861 * lam2 / (lam2 - .02641)
               + 35.65066 * lam2 / (lam2 - 1268.24708))
    elif model == "lorentz":
        # Transparent, zero-damping approximation used ONLY for sensitivity.
        eps = 6.7 * (970.0 ** 2 - v ** 2) / (797.0 ** 2 - v ** 2)
    elif model == "constant":
        eps = np.full_like(v, 2.55 ** 2)
    else:
        raise ValueError(model)
    return np.sqrt(eps) * scale


def select(data, window=MAIN_WINDOW, exclude=True, stride=1):
    out = []
    for a in data:
        mask = (a[:, 0] >= window[0]) & (a[:, 0] <= window[1])
        if exclude:
            mask &= ~((a[:, 0] > EXCLUDED[0]) & (a[:, 0] < EXCLUDED[1]))
        out.append(a[mask][::stride].copy())
    return out


def matrix(v, d, phase, angle, q, window, degree, envelope_degree, model, scale):
    t = 2 * (v - window[0]) / (window[1] - window[0]) - 1
    phase_v = 4 * np.pi * d * 1e-4 * v * np.sqrt(
        index(v, model, scale) ** 2 - np.sin(np.deg2rad(angle)) ** 2) + phase
    cosine = np.cos(phase_v)
    # Sum_{k>=1} q^(k-1) cos(k*phase); q=0 is exactly the two-beam model.
    oscillation = (cosine - q) / (1 - 2 * q * cosine + q * q)
    baseline = chebvander(t, degree)
    amplitude_basis = np.stack([t ** k for k in range(envelope_degree + 1)], axis=1)
    return np.column_stack((baseline, amplitude_basis * oscillation[:, None]))


def fit(sets, angles=ANGLES, window=MAIN_WINDOW, model="wang", scale=1.0,
        degree=3, envelope_degree=2, airy=False, initial=None, multistart=False):
    ns = len(sets)
    bounds = ([5.0] + [-100.0] * ns + ([-.45] * ns if airy else []),
              [10.0] + [100.0] * ns + ([.45] * ns if airy else []))

    def residual(p, detail=False):
        residuals, coefficients, predictions = [], [], []
        for i, (a, angle) in enumerate(zip(sets, angles)):
            dm = matrix(a[:, 0], p[0], p[1 + i], angle,
                        p[1 + ns + i] if airy else 0.0, window, degree,
                        envelope_degree, model, scale)
            beta = np.linalg.lstsq(dm, a[:, 1], rcond=None)[0]
            pred = dm @ beta
            residuals.append(pred - a[:, 1])
            coefficients.append(beta)
            predictions.append(pred)
        joined = np.concatenate(residuals)
        return (joined, coefficients, predictions) if detail else joined

    if initial is None:
        initial = [7.4] + [-3.3, -3.75][:ns] + ([0.0] * ns if airy else [])
    starts = [initial]
    if multistart:
        starts += [[float(d)] + [p] * ns + ([0.0] * ns if airy else [])
                   for d in np.arange(6.6, 8.01, .2) for p in (0.0, 1.5)]
    solutions = [least_squares(residual, p, bounds=bounds, max_nfev=120,
                              ftol=1e-10, xtol=1e-10, gtol=1e-10) for p in starts]
    best = min(solutions, key=lambda r: float(r.fun @ r.fun))
    # Resolve the arbitrary sign of the fundamental amplitude so reported q
    # has the same convention for both angles and for all runs.
    _, initial_coef, _ = residual(best.x, detail=True)
    for i, beta in enumerate(initial_coef):
        if beta[degree + 1] < 0:
            best.x[1 + i] += np.pi
            if airy:
                best.x[1 + ns + i] *= -1
        best.x[1 + i] = (best.x[1 + i] + np.pi) % (2 * np.pi) - np.pi
    r, coef, pred = residual(best.x, detail=True)
    n = len(r)
    k = len(best.x) + sum(len(b) for b in coef)
    sse = float(r @ r)
    sst = sum(float(np.sum((a[:, 1] - np.mean(a[:, 1])) ** 2)) for a in sets)
    return dict(d_um=float(best.x[0]), parameters=best.x, coefficients=coef,
                predictions=pred, residuals=[p - a[:, 1] for p, a in zip(pred, sets)],
                rmse_pp=float(np.sqrt(sse / n)), sse_pp2=sse, n=n, k=k,
                naive_bic=float(n * np.log(sse / n) + k * np.log(n)),
                r2=1-sse/sst, success=bool(best.success), window=list(window),
                model=model, scale=scale, degree=degree,
                envelope_degree=envelope_degree, airy=airy, angles=list(angles),
                starts=len(starts))


def serial_fit(f):
    omit = {"predictions", "residuals"}
    return {k: ([b.tolist() for b in v] if k == "coefficients" else
                v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in f.items() if k not in omit}


def predict(a, i, f):
    p, ns = f["parameters"], len(f["angles"])
    dm = matrix(a[:, 0], p[0], p[1+i], f["angles"][i],
                p[1+ns+i] if f["airy"] else 0.0, f["window"], f["degree"],
                f["envelope_degree"], f["model"], f["scale"])
    return dm @ f["coefficients"][i]


def baselines(data):
    result = []
    for i, a in enumerate(data):
        x, y = a[:, 0], a[:, 1]
        mask = (x >= MAIN_WINDOW[0]) & (x <= MAIN_WINDOW[1])
        xx, yy = x[mask], y[mask]
        t = (xx - xx.mean()) / np.ptp(xx)
        detrended = yy - np.polynomial.polynomial.polyval(
            t, np.polynomial.polynomial.polyfit(t, yy, 3))
        fft_size = 2 ** 18
        freq = np.fft.rfftfreq(fft_size, np.median(np.diff(xx)))
        power = np.abs(np.fft.rfft(detrended * np.hanning(len(xx)), fft_size)) ** 2
        search = (freq >= .002) & (freq <= .008)
        peak_frequency = float(freq[search][np.argmax(power[search])])
        angle = ANGLES[i]
        dfft = 1e4 * peak_frequency / (2 * np.sqrt(2.55 ** 2 - np.sin(np.deg2rad(angle)) ** 2))
        # Extrema are a deliberately independent initialization. Smoothing is
        # 31 samples = 14.95 cm^-1; no smoothing is used in the main regression.
        smooth = savgol_filter(yy, 31, 3)
        peaks, _ = find_peaks(smooth, distance=350, prominence=.12)
        vp = xx[peaks]
        # Reject peaks in the predeclared local atmospheric feature interval.
        vp = vp[~((vp > 2300) & (vp < 2400))]
        if len(vp) >= 3:
            # Removing an anomalous peak may remove a fringe: reconstruct the
            # integer order using FFT period so gaps cannot be counted as one.
            orders = np.r_[0, np.cumsum(np.maximum(1, np.rint(np.diff(vp) * peak_frequency)).astype(int))]
            z = vp * np.sqrt(index(vp) ** 2 - np.sin(np.deg2rad(angle)) ** 2)
            slope = np.polyfit(orders, z, 1)[0]
            dpeaks = 1e4 / (2 * slope)
        else:
            orders, dpeaks = [], None
        result.append(dict(attachment=i+1, window=list(MAIN_WINDOW),
                           fft_frequency_cm=peak_frequency,
                           fft_period_cm_inverse=1/peak_frequency,
                           fft_constant_n_d_um=float(dfft),
                           raw_fft_bin_resolution_cm=float(1/(xx[-1]-xx[0])),
                           zero_padding_is_not_independent_resolution=True,
                           maxima_cm_inverse=vp.tolist(), fringe_orders=list(map(int, orders)),
                           maxima_wang_d_um=float(dpeaks) if dpeaks is not None else None))
        pd.DataFrame(dict(frequency_cm=freq[search], power=power[search])).to_csv(
            ROOT / "results" / f"sic_fft_{i+1}.csv", index=False)
    return result


def cross_validate(sets, baseline_fit, airy_fit):
    """Five interleaved folds of contiguous 50 cm^-1 blocks; no random rows."""
    records = []
    for fold in range(5):
        train, test = [], []
        for a in sets:
            ids = np.floor((a[:, 0] - MAIN_WINDOW[0]) / 50.0).astype(int) % 5
            train.append(a[ids != fold]); test.append(a[ids == fold])
        row = dict(fold=fold, n_test=sum(len(a) for a in test))
        for name, ref in [("two_beam", baseline_fit), ("airy", airy_fit)]:
            f = fit(train, airy=ref["airy"], initial=ref["parameters"])
            rr = np.concatenate([predict(a, i, f) - a[:, 1] for i, a in enumerate(test)])
            row[name + "_sse"] = float(rr @ rr)
            row[name + "_rmse_pp"] = float(np.sqrt(np.mean(rr ** 2)))
            row[name + "_d_um"] = f["d_um"]
        records.append(row)
    n = sum(r["n_test"] for r in records)
    summary = {name+"_rmse_pp":float(np.sqrt(sum(r[name+"_sse"] for r in records)/n))
               for name in ("two_beam", "airy")}
    return dict(block_width_cm_inverse=50.0, folds=records, pooled=summary)


def bootstrap(data, iterations):
    """Paired circular moving-block residual bootstrap on every fifth row.

    The same block indices are used at both angles to preserve cross-spectrum
    residual dependence. It quantifies only conditional residual variation.
    """
    sets = select(data, stride=5)
    base = fit(sets)
    airy = fit(sets, airy=True, initial=np.r_[base["parameters"], 0, 0])
    rng = np.random.default_rng(SEED)
    n = len(sets[0])
    block_rows = int(round(50 / np.median(np.diff(sets[0][:, 0]))))
    # fit() stores prediction-minus-observation for optimization. Resampling
    # requires observation-minus-prediction before adding errors to predictions.
    rmat = -np.stack(base["residuals"], axis=1)
    rmat -= rmat.mean(axis=0)
    arr = []
    for b in range(iterations):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_rows)))
        ids = np.concatenate([(s + np.arange(block_rows)) % n for s in starts])[:n]
        pseudo = [np.column_stack((a[:, 0], base["predictions"][i] + rmat[ids, i]))
                  for i, a in enumerate(sets)]
        f = fit(pseudo, initial=base["parameters"])
        arr.append([b, f["d_um"]])
    values = np.asarray(arr)[:, 1]
    pd.DataFrame(arr, columns=["iteration", "d_um"]).to_csv(
        ROOT / "results" / "sic_bootstrap.csv", index=False)
    return dict(seed=SEED, iterations=iterations, stride=5, block_rows=block_rows,
                block_width_approx_cm_inverse=50, center_d_um=base["d_um"],
                mean_d_um=float(np.mean(values)), std_d_um=float(np.std(values, ddof=1)),
                percentile95_d_um=np.quantile(values, [.025, .975]).tolist(),
                interpretation="Conditional moving-block residual range, not an absolute accuracy interval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=120)
    args = parser.parse_args()
    if args.bootstrap < 20:
        raise ValueError("Use at least 20 bootstrap resamples.")
    (ROOT / "results").mkdir(exist_ok=True)
    data, quality = [], []
    for i in (1, 2):
        path = ROOT / "data" / f"附件{i}.xlsx"
        raw = pd.read_excel(path).to_numpy(dtype=float)
        if raw.shape[1] != 2:
            raise ValueError(f"Expected two columns: {path.name}")
        finite = np.isfinite(raw).all(axis=1)
        zero = (raw[:, 1] == 0)
        a = raw[finite & ~zero]
        if not np.all(np.diff(a[:, 0]) > 0):
            raise ValueError("Wavenumber must increase without duplicates")
        data.append(a)
        quality.append(dict(attachment=i, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            rows_raw=len(raw), rows_used_after_cleaning=len(a),
                            nonfinite_rows=int(np.sum(~finite)), zero_rows=int(np.sum(zero)),
                            excluded_zero_wavenumbers=raw[zero, 0].tolist(),
                            wavenumber_range=[float(raw[0, 0]), float(raw[-1, 0])],
                            median_spacing=float(np.median(np.diff(a[:, 0]))),
                            reflectance_min=float(np.min(a[:, 1])),
                            reflectance_max=float(np.max(a[:, 1])),
                            over_100_count=int(np.sum(a[:, 1] > 100))))
    sets = select(data)
    # A coarse multi-start on every third row chooses the basin. The reported
    # result and residuals are then fitted to every retained measured row.
    coarse = fit(select(data, stride=3), multistart=True)
    two = fit(sets, initial=coarse["parameters"])
    airy = fit(sets, airy=True, initial=np.r_[two["parameters"], 0., 0.])
    independent = [fit([a], angles=[angle], initial=[two["d_um"], two["parameters"][1+i]])
                   for i, (a, angle) in enumerate(zip(sets, ANGLES))]
    print("SiC reference:", two["d_um"], "um; RMSE", two["rmse_pp"], "pp", flush=True)
    print("SiC Airy:", airy["d_um"], "um; q", airy["parameters"][-2:], flush=True)

    sensitivity = []
    for window in [(2000.,3200.), (2100.,3300.), (2200.,3300.), (2000.,3000.), (2000.,3100.)]:
        f = fit(select(data, window), window=window, initial=two["parameters"])
        sensitivity.append(dict(kind="window", **serial_fit(f)))
    for window in [(1800.,3300.), (1900.,3300.)]:
        f = fit(select(data, window), window=window, initial=two["parameters"])
        sensitivity.append(dict(kind="extrapolation_beyond_valid_5um_not_reference", **serial_fit(f)))
    for degree, env in [(2,2),(4,2),(3,1)]:
        f = fit(sets, degree=degree, envelope_degree=env, initial=two["parameters"])
        sensitivity.append(dict(kind="background_envelope", **serial_fit(f)))
    for model, scale in [("constant",1), ("lorentz",1), ("wang",.99), ("wang",1.01)]:
        f = fit(sets, model=model, scale=scale, initial=two["parameters"], multistart=True)
        sensitivity.append(dict(kind="index_assumption", **serial_fit(f)))
    f = fit(select(data, exclude=False), initial=two["parameters"])
    sensitivity.append(dict(kind="retain_2300_2400", **serial_fit(f)))

    comparison = baselines(data)
    cv = cross_validate(sets, two, airy)
    boot = bootstrap(data, args.bootstrap)
    for i, a in enumerate(sets):
        coeff = two["coefficients"][i]
        t = 2 * (a[:, 0] - MAIN_WINDOW[0]) / (MAIN_WINDOW[1]-MAIN_WINDOW[0]) - 1
        baseline = chebvander(t, two["degree"]) @ coeff[:two["degree"]+1]
        amp = sum(coeff[two["degree"]+1+j] * t ** j for j in range(two["envelope_degree"]+1))
        pd.DataFrame(dict(wavenumber_cm_inverse=a[:, 0], reflectance_percent=a[:, 1],
                          index_wang=index(a[:, 0]), two_beam_percent=two["predictions"][i],
                          airy_percent=airy["predictions"][i], baseline_percent=baseline,
                          amplitude_percent=amp, residual_two_beam_pp=-two["residuals"][i],
                          residual_airy_pp=-airy["residuals"][i])).to_csv(
            ROOT / "results" / f"sic_prediction_{i+1}.csv", index=False)
    group_v = np.array([2000.,2500.,3300.])
    h = .01
    group_n = index(group_v) + group_v*(index(group_v+h)-index(group_v-h))/(2*h)
    result = dict(
        input_quality=quality, reference_window_cm_inverse=list(MAIN_WINDOW),
        excluded_local_band_cm_inverse=list(EXCLUDED),
        reference_index_assumption="Wang2013 4H-SiC ordinary ray Sellmeier, conditional on unspecified sample polytype/polarization",
        index_reference_url="https://refractiveindex.info/?shelf=main&book=SiC&page=Wang-4H-o",
        original_index_paper_doi="10.1002/lpor.201300068",
        reference_index_sample=dict(wavenumber=group_v.tolist(), n=index(group_v).tolist(),
                                    group_index=group_n.tolist()),
        two_beam=serial_fit(two), airy=serial_fit(airy),
        independent_angles=[serial_fit(f) for f in independent],
        initializers=comparison, sensitivity=sensitivity,
        cross_validation=cv, conditional_bootstrap=boot,
        correction_d_um=airy["d_um"]-two["d_um"],
        correction_percent=100*(airy["d_um"]-two["d_um"])/two["d_um"],
        multi_beam_notes=[
            "q is an effective signed harmonic-shape coefficient, not a uniquely measured interface reflectivity.",
            "The model at q=0 is nested exactly in the effective Airy model.",
            "The slowly varying baseline and amplitude absorb calibration and Fresnel envelopes.",
            "Raw-sample BIC is descriptive: closely spaced residuals are not independent.",
            "Use blocked predictive error and thickness stability rather than only naive BIC for interpretation."],
        limitations=[
            "4H ordinary-ray refractive index is assumed; sample polytype, axis, doping and temperature are unspecified.",
            "The fitted quantity becomes geometric thickness only conditional on the specified n(wavenumber).",
            "The 2000-3300 cm^-1 reference window avoids the SiC Reststrahlen band and stays inside both the database formula range (0.4047-5 um) and the paper's reported measured MIR band (3-5 um).",
            "The 2300-2400 cm^-1 local anomaly is excluded in both spectra; attribution to atmospheric CO2 is plausible, not experimentally established here.",
            "A measured reflectance above 100% indicates calibration inconsistency; it is retained in raw data and lies outside the reference fitting band.",
            "Residual bootstrap does not capture refractive-index, angular calibration, substrate-phase or instrumental systematic errors.",
            "A small harmonic correction in this selected transparent band does not prove that multiple reflections are absent throughout the full spectrum."])
    out = ROOT / "results" / "sic_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", out, flush=True)
    print("Independent angles", [f["d_um"] for f in independent], flush=True)
    print("CV", cv["pooled"], "bootstrap95", boot["percentile95_d_um"], flush=True)


if __name__ == "__main__":
    main()

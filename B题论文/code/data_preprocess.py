"""Audit source spectra without imputing endpoint zeros or clipping reflectance."""
from pathlib import Path
import hashlib
import json
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]


def prepare_data():
    """Keep original measurements and emit a deterministic quality report."""
    reports = []
    (ROOT / 'results').mkdir(exist_ok=True)
    for i in range(1, 5):
        path = ROOT / 'data' / f'附件{i}.xlsx'
        workbook = load_workbook(path, read_only=True, data_only=True)
        values = list(workbook.active.values)
        a = np.asarray(values[1:], dtype=float)
        workbook.close()
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f'{path.name}: expected two numeric columns')
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{path.name}: nonfinite measurements require review')
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError(f'{path.name}: wavenumbers must increase strictly')
        np.savetxt(ROOT / 'data' / f'spectrum_{i}.csv', a, delimiter=',',
                   header='wavenumber_cm_inv,reflectance_percent', comments='', fmt='%.10g')
        reports.append({
            'id': i, 'material': 'SiC' if i < 3 else 'Si',
            'angle_degree': 10 if i % 2 else 15,
            'rows': len(a), 'missing_cells': int(np.isnan(a).sum()),
            'duplicate_wavenumbers': len(a) - len(np.unique(a[:, 0])),
            'wavenumber_min': float(a[:, 0].min()),
            'wavenumber_max': float(a[:, 0].max()),
            'step_median': float(np.median(np.diff(a[:, 0]))),
            'reflectance_min': float(a[:, 1].min()),
            'reflectance_max': float(a[:, 1].max()),
            'zero_count': int((a[:, 1] == 0).sum()),
            'over100_count': int((a[:, 1] > 100).sum()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (ROOT / 'results' / 'data_quality.json').write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    return reports


if __name__ == '__main__':
    for row in prepare_data():
        print(f"Spectrum {row['id']}: {row['rows']} points, "
              f"{row['zero_count']} zero, {row['over100_count']} above 100%")

"""Reproduce problem-B results, figures, tables and checks without PDF/Word."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=120)
    parser.add_argument('--figures-only', action='store_true',
                        help='Regenerate figures/tables and verify saved numerical results.')
    args = parser.parse_args()
    if args.bootstrap < 20:
        raise ValueError('Use at least 20 resamples; the paper uses 120.')
    optional = ROOT / 'scratch' / 'sic_deps'
    env = os.environ.copy()
    if optional.is_dir():
        env['PYTHONPATH'] = str(optional) + os.pathsep + env.get('PYTHONPATH', '')
        sys.path.insert(0, str(optional))
    env['PYTHONIOENCODING'] = 'utf-8'
    stages = []
    if not args.figures_only:
        stages += [('data_preprocess.py', []),
                   ('sic_analysis.py', ['--bootstrap', str(args.bootstrap)]),
                   ('silicon_analysis.py', ['--bootstrap', str(args.bootstrap)]),
                   ('joint_sensitivity.py', [])]
    stages += [('common_figures.py', []), ('result_figures.py', []), ('verify_project.py', [])]
    records = []
    started = time.time()
    for script, extra in stages:
        print(f'Running {script}', flush=True)
        tic = time.perf_counter()
        proc = subprocess.run([sys.executable, str(ROOT / 'code' / script), *extra],
                              cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        print(proc.stdout, end='', flush=True)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end='', flush=True)
        records.append(dict(script=script, elapsed_seconds=time.perf_counter()-tic,
                            exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
        if proc.returncode:
            (ROOT / 'results' / 'failed_run.json').write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            raise SystemExit(proc.returncode)
    versions = {name: importlib.metadata.version(name)
                for name in ['numpy', 'pandas', 'openpyxl', 'scipy', 'matplotlib']}
    report = dict(started_unix=started, total_seconds=time.time()-started,
                  python=platform.python_version(), platform=platform.platform(),
                  processor=platform.processor(), logical_processors=os.cpu_count(),
                  package_versions=versions, figures_only=args.figures_only,
                  bootstrap_replicates=args.bootstrap, stages=records,
                  creates_pdf=False, creates_word=False)
    filename = 'figure_run.json' if args.figures_only else 'run_log.json'
    (ROOT / 'results' / filename).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Complete: {filename}; no PDF or Word generated.')


if __name__ == '__main__':
    main()

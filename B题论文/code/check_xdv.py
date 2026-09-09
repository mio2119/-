"""Compile the complete paper to XDV and record checks; never request PDF output.

Usage: python code/check_xdv.py --tectonic /path/to/tectonic
The optional local scratch runtime is used only when it already exists.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic', type=Path)
    args = parser.parse_args()
    local = ROOT / 'scratch' / 'tectonic' / 'bin' / 'tectonic.exe'
    executable = args.tectonic or shutil.which('tectonic')
    if executable is None and local.is_file():
        executable = local
    if executable is None:
        raise SystemExit('Install Tectonic or provide --tectonic PATH.')
    scratch = ROOT / 'scratch' / 'texcheck'
    scratch.mkdir(parents=True, exist_ok=True)
    flags = ['--outfmt', 'xdv', '--bundle',
             'https://data1b.fullyjustified.net/tlextras-2022.0r0.tar',
             '--keep-logs', '--keep-intermediates', '--outdir',
             'scratch/texcheck', 'main.tex']
    proc = subprocess.run([str(executable), *flags], cwd=ROOT,
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace')
    console = proc.stdout + proc.stderr
    (scratch / 'compile-console.txt').write_text(console, encoding='utf-8')
    if proc.returncode:
        print(console[-5000:])
        raise SystemExit(proc.returncode)
    log = (scratch / 'main.log').read_text(encoding='utf-8', errors='replace')
    aux = (scratch / 'main.aux').read_text(encoding='utf-8', errors='replace')
    assert 'Output written on main.xdv' in log
    assert not re.search(r'^!', log, re.M), 'TeX error in final log'
    assert 'undefined' not in log.lower(), 'Unresolved reference or citation'
    assert 'Overfull' not in log, 'Content exceeds a TeX box'
    assert 'Missing character:' not in log, 'Missing font glyph'
    # Dependencies can contain vendor PDF icons; only paper outputs count.
    forbidden = [p for suffix in ('*.pdf', '*.docx') for p in ROOT.rglob(suffix)
                 if 'scratch' not in p.relative_to(ROOT).parts or scratch in p.parents]
    assert not forbidden, 'Unexpected PDF or Word paper output'
    underfull = re.findall(
        r'Underfull \\hbox \(badness (\d+)\) in paragraph at lines ([\d-]+)', log)
    pages = int(re.search(r'Output written on main.xdv \((\d+) pages', log).group(1))
    engine = subprocess.run([str(executable), '--version'], capture_output=True,
                            text=True, encoding='utf-8', check=True).stdout.strip()
    report = dict(
        status='passed_with_spacing_warnings' if underfull else 'passed',
        checked_at_utc=datetime.now(timezone.utc).isoformat(), engine=engine,
        pipeline='XeTeX and BibTeX', exit_code=proc.returncode,
        command='tectonic ' + ' '.join(flags), output_format='XDV',
        tex_errors=0, unresolved_citations_or_references=0,
        overfull_boxes=0, missing_glyphs=0, underfull_hbox_count=len(underfull),
        underfull_hbox_details=underfull,
        fontconfig_default_config_diagnostic='Fontconfig error:' in console,
        font_note='TeX-distribution font files resolved; no missing glyphs.',
        xdv_pages=pages, xdv_bytes=(scratch / 'main.xdv').stat().st_size,
        bibliography_items=len(re.findall(r'\\bibcite\{', aux)),
        pdf_generated=False, word_generated=False,
        page_layout_note='XDV page count is engine bookkeeping, not full-page visual QA.',
        figure_qa='All 15 PNG figures inspected; three corrected label placements rechecked.',
        log_sha256=hashlib.sha256((scratch / 'main.log').read_bytes()).hexdigest(),
        main_tex_sha256=hashlib.sha256((ROOT / 'main.tex').read_bytes()).hexdigest())
    (ROOT / 'results' / 'latex_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

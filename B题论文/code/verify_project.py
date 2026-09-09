"""Check physical predictions, data provenance and the complete LaTeX source tree."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
from sic_analysis import predict as predict_sic
from silicon_analysis import predicted as predict_si, numerical_self_checks

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / 'results' / name).read_text(encoding='utf-8'))


def macro_args(text, macro, count):
    """Read brace-balanced arguments, including captions with nested math groups."""
    output=[]
    for match in re.finditer(r'\\'+re.escape(macro)+r'\s*\{',text):
        pos=match.end()-1;args=[]
        for _ in range(count):
            while pos<len(text) and text[pos].isspace():pos+=1
            if pos>=len(text) or text[pos]!='{':break
            start=pos+1;depth=1;pos+=1
            while pos<len(text) and depth:
                if text[pos] not in '{}':pos+=1;continue
                if pos and text[pos-1]=='\\':pos+=1;continue
                depth += 1 if text[pos]=='{' else -1
                pos+=1
            args.append(text[start:pos-1])
        if len(args)==count:output.append(args)
    return output


def source_tree():
    visited=[]
    def expand(path):
        if path in visited:raise AssertionError(f'Repeated source input: {path.name}')
        visited.append(path)
        raw=path.read_text(encoding='utf-8')
        assert all(ord(c)>=32 or c in '\n\r\t' for c in raw), f'Invalid control character: {path.name}'
        text='\n'.join(re.split(r'(?<!\\)%',line)[0] for line in raw.splitlines())
        for name in re.findall(r'\\input\{([^}]+)\}',text):
            p=ROOT/name
            if not p.suffix:p=p.with_suffix('.tex')
            assert p.is_file(),f'Missing source {p}'
            text=text.replace('\\input{'+name+'}',expand(p))
        return text
    text=expand(ROOT/'main.tex')
    level=0
    for char in re.sub(r'\\[{}%]','',text):
        if char=='{':level+=1
        if char=='}':level-=1
        assert level>=0,'Unbalanced closing brace'
    assert level==0,'Unbalanced opening brace'
    stack=[]
    for direction,env in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
        if direction=='begin':stack.append(env)
        else:assert stack and stack.pop()==env,f'Environment mismatch: {env}'
    assert not stack,'Unclosed environment'
    figures=macro_args(text,'paperfigure',4)
    figures=[f for f in figures if '#' not in f[0]]
    labels=[s for s in re.findall(r'\\label\{([^}]+)\}',text) if '#' not in s]
    labels += [f[3] for f in figures]
    assert len(labels)==len(set(labels)),'Duplicate labels'
    refs=[s for s in re.findall(r'\\(?:ref|eqref|cref)\{([^}]+)\}',text) if '#' not in s]
    assert set(refs)<=set(labels),f'Unresolved labels: {set(refs)-set(labels)}'
    for filename,_,width,_ in figures:
        p=ROOT/filename
        assert p.is_file(),f'Missing figure {filename}'
        with Image.open(p) as im:
            assert im.width>1000 and im.height>500,f'Low-resolution figure {filename}'
        assert float(width)<=.80,f'Oversized figure width: {filename}'
    cites=[]
    for group in re.findall(r'\\cite(?:p|t)?(?:\[[^\]]*\])?\{([^}]+)\}',text):
        cites.extend(group.split(','))
    bib=(ROOT/'references.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    assert set(cites)<=keys,f'Undefined citations {set(cites)-keys}'
    for term in ['TODO','待补','占位','尚未包含','骨架']:
        assert term not in text,f'Unfinished content: {term}'
    assert '7.38431' in text and '3.40521' in text,'Main numerical conclusions missing'
    assert not list(ROOT.glob('*.pdf')) and not list(ROOT.glob('*.docx'))
    return dict(source_files=len(visited),figures_referenced=len(figures),
                tables=len(re.findall(r'\\begin\{table\}',text)),
                cited_references=len(set(cites)),uncited_bibliography=sorted(keys-set(cites)),
                unique_labels=len(labels),cross_references=len(refs),
                source_checks='passed',compile_status='not_verified_by_this_static_check')


def main():
    quality=read_json('data_quality.json')
    for row in quality:
        p=ROOT/'data'/f"附件{row['id']}.xlsx"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],'Source changed'
    sic=read_json('sic_results.json');si=read_json('silicon_results.json')
    discrepancies={}
    for i in [1,2]:
        a=pd.read_csv(ROOT/'results'/f'sic_prediction_{i}.csv')
        z=a[['wavenumber_cm_inverse','reflectance_percent']].to_numpy()
        for key,col in [('two_beam','two_beam_percent'),('airy','airy_percent')]:
            f=sic[key]
            yh=predict_sic(z,i-1,f)
            delta=float(np.max(abs(yh-a[col].to_numpy())))
            assert delta<1e-8,f'SiC stored prediction mismatch: {delta}'
            discrepancies[f'sic_{i}_{key}']=delta
    for i in [3,4]:
        a=pd.read_csv(ROOT/'results'/f'silicon_prediction_{i}.csv')
        z=a[['sigma_cm1','R_pct']].to_numpy()
        for key,kind,col in [('main','multi','multi_pct'),('two_ray','two','two_pct')]:
            f=si[key]
            yh=predict_si([z],[10 if i==3 else 15],f['parameters'],
                          [f['calibration_gain_offset'][i-3]],kind)[0]
            delta=float(np.max(abs(yh-a[col].to_numpy())))
            assert delta<1e-8,f'Si stored prediction mismatch: {delta}'
            discrepancies[f'si_{i}_{key}']=delta
    report=dict(source_sha256='four original workbooks unchanged',
                maximum_prediction_discrepancies_pp=discrepancies,
                physical_limit_checks=numerical_self_checks(),latex=source_tree(),
                note='Numerical/source validation does not establish full page-layout quality.')
    (ROOT/'results'/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

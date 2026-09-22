#!/usr/bin/env python3
"""Verify all distributed file hashes and Python syntax without running research drivers."""
from __future__ import annotations
import argparse,ast,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(repo:Path):
    manifest=json.loads((repo/'metadata/release_checksums.json').read_text())
    failures=[]
    for row in manifest['files']:
        p=repo/row['path']
        if not p.is_file():failures.append('Missing: '+row['path']);continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:failures.append('Changed: '+row['path'])
        if p.suffix=='.py':
            try:ast.parse(p.read_text(encoding='utf-8'),filename=row['path'])
            except SyntaxError:failures.append('Syntax: '+row['path'])
        if p.suffix.lower() in {'.npz','.npy','.whl','.ttf','.otf'}:failures.append('Unexpected binary asset: '+row['path'])
    allowed={row['path'] for row in manifest['files']} | {'metadata/release_checksums.json'}
    skip={'outputs','.venv','venv','.git','__pycache__','.ipynb_checkpoints','.pytest_cache'}
    for path in repo.rglob('*'):
        if path.is_file():
            rel=path.relative_to(repo)
            if not any(part in skip for part in rel.parts) and rel.as_posix() not in allowed:
                failures.append('Unlisted file: '+rel.as_posix())
    result={'verified_files':len(manifest['files']),'failures':failures,'passed':not failures}
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,default=ROOT);a=p.parse_args()
    sys.exit(0 if run(a.repository)['passed'] else 1)

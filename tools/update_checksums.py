#!/usr/bin/env python3
"""Explicit maintainer operation: update distribution hashes after approved edits."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'outputs','.venv','venv','.git','__pycache__','.ipynb_checkpoints'}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--confirm-update',action='store_true');a=p.parse_args()
if not a.confirm_update:p.exit(2,'Pass --confirm-update only after reviewing intended changes.\n')
rows=[]
for f in sorted(ROOT.rglob('*')):
    if f.is_file() and not any(x in SKIP for x in f.relative_to(ROOT).parts) and f!=ROOT/'metadata/release_checksums.json':
        rows.append({'path':f.relative_to(ROOT).as_posix(),'size_bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(ROOT/'metadata/release_checksums.json').write_text(json.dumps({'scope':'Distributed files excluding this self-referential manifest and generated/local directories.','files':rows},indent=2))
print(f'Updated {len(rows)} distribution checksums. Source/result provenance manifests are not rewritten.')

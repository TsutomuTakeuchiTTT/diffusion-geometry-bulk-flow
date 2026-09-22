#!/usr/bin/env python3
"""Redraw Paper I Figure 6 from bundled saved CSVs using its recovered plot script.

No fit is run. Output is a fresh rendering, not a claim of pixel identity with
historic publication files. The original source's output/summary conventions
are preserved. The output directory must not already exist.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='research_sources/paper_i/paperI_figure4_mock1_scale_dependence_converged.py'

def run(repo:Path,output:Path):
    repo=repo.resolve();output=output.resolve()
    if output.exists():raise FileExistsError('Choose a new output directory.')
    saved=repo/'results/saved'
    # Read-only saved data are kept separate from all generated files.
    if output==saved or saved in output.parents:raise ValueError('Output cannot be inside archived saved results.')
    expected=next(r['release_sha256'] for r in json.loads((repo/'metadata/source_manifest.json').read_text())['sources'] if r['path']==REL)
    if hashlib.sha256((repo/REL).read_bytes()).hexdigest()!=expected:raise ValueError('Plot source fingerprint changed.')
    env=os.environ.copy()
    env.update({'MPLBACKEND':'Agg','PAPER1_ROOT_DIR':str(saved),'PAPER1_FIGURE4_SHOW_PLOTS':'0','PAPER1_FIGURE4_OUTPUT':str(output)})
    subprocess.run([sys.executable,str(repo/REL)],cwd=repo,env=env,check=True)
    return output
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=Path('outputs/paperI_figure06'));a=p.parse_args()
    try:run(a.repository,a.output)
    except (OSError,ValueError,subprocess.CalledProcessError) as e:p.exit(2,str(e)+'\n')

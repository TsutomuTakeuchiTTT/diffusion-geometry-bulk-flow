#!/usr/bin/env python3
"""Recompute 27 Paper-II capacity decisions from saved curves.

No regression/eigensolver/bootstrap run. Plateau routines are loaded from the
corresponding archived research source, without executing its production driver.
The raw-minimum and one-SE computations use the full sorted capacity grid.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from original_function_loader import load_functions

ROOT=Path(__file__).resolve().parents[1]
CAPACITY_FIELDS=('validation_minimum_basis','one_standard_error_basis',
                 'practical_plateau_basis','combined_practical_basis')

def choices(basis,mean,sem,plateau):
    b=np.asarray(basis,dtype=int);m=np.asarray(mean,dtype=float);s=np.asarray(sem,dtype=float)
    if len(b)<3 or len(np.unique(b))!=len(b) or np.any(np.diff(b)<=0):
        raise ValueError('Require a strictly increasing capacity grid.')
    if m.shape!=b.shape or s.shape!=b.shape or not np.all(np.isfinite(m)) or not np.all(np.isfinite(s)) or np.any(s<0):
        raise ValueError('Invalid mean/standard-error arrays.')
    j=int(np.argmin(m))
    one=int(b[np.flatnonzero(m<=m[j]+s[j])[0]])
    plat=plateau(b,m)
    if plat is None or not np.isfinite(plat):raise ValueError('No plateau found; do not fabricate convergence.')
    return dict(zip(CAPACITY_FIELDS,[int(b[j]),one,int(plat),max(one,int(plat))]))

def run(repo:Path,output:Path):
    if output.exists():raise FileExistsError('Use a fresh output directory.')
    saved=repo/'results/saved';src=repo/'research_sources/paper_ii'
    manifests=json.loads((repo/'metadata/source_manifest.json').read_text())['sources']
    import hashlib
    for row in manifests:
        if hashlib.sha256((repo/row['path']).read_bytes()).hexdigest()!=row['release_sha256']:
            raise ValueError('Source fingerprint changed: '+row['path'])
    cases=[]
    for mock in [2,3]:
        stem=f'mock{mock}_radial_potential_highmode_seed_robustness'
        filename=stem+('_v1.py' if mock==2 else '_v1_fixed_panel_d.py')
        ns,_=load_functions(src/filename,['_practical_plateau'],[],
             {'np':np,'math':math,'PRACTICAL_PLATEAU_CONSECUTIVE_STEPS':2,'PRACTICAL_PLATEAU_RELATIVE_THRESHOLD':.02})
        plateau=lambda b,m,ns=ns:ns['_practical_plateau'](b,m)['plateau_basis']
        cases.append((stem+'_v1',stem,['loss_name'],'weighted_validation_nrmse_radial_mean',
                      'weighted_validation_nrmse_radial_sem',plateau,3))
    for name in ['measurement_noise_stage1b_seed_robustness','heteroscedastic_noise_stage2']:
        stem='mock1_potential_flow_'+name
        ns,_=load_functions(src/(stem+'_v1.py'),['practical_plateau_basis'],[],{'np':np})
        plateau=lambda b,m,ns=ns:ns['practical_plateau_basis'](b,m,514,.02,2)[0]
        noise=name.startswith('measurement')
        cases.append((stem+'_v1',stem,['noise_fraction'] if noise else ['working_model','noise_fraction'],
             'validation_observed_nrmse_mean' if noise else 'validation_mean',
             'validation_observed_nrmse_sem_between_seeds' if noise else 'validation_sem_between_seeds',plateau,514))
    records=[]
    for dirname,stem,keys,mc,sc,plateau,floor in cases:
        curve=pd.read_csv(saved/dirname/(stem+'_capacity_aggregate.csv'))
        selection=pd.read_csv(saved/dirname/(stem+'_capacity_selection.csv'))
        for vals,group in curve.groupby(keys,sort=True):
            if not isinstance(vals,tuple):vals=(vals,)
            group=group.sort_values('n_basis')
            sel=selection
            for key,val in zip(keys,vals):sel=sel.loc[sel[key]==val]
            if len(sel)!=1:raise ValueError('Ambiguous selection row.')
            got=choices(group.n_basis,group[mc],group[sc],plateau)
            expected={key:int(sel.iloc[0][key]) for key in CAPACITY_FIELDS}
            records.append({'experiment':stem,'group':{key:val.item() if isinstance(val,np.generic) else val for key,val in zip(keys,vals)},
                            'plateau_search_floor':floor,'recomputed':got,'saved':expected,'match':got==expected})
    output.mkdir(parents=True)
    result={'scope':'Recomputed from saved aggregate curves with original plateau functions. Not a new fit.',
            'groups':len(records),'all_match':all(x['match'] for x in records),'details':records}
    (output/'capacity_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame([{'experiment':r['experiment'],**r['group'],**r['recomputed'],'match':r['match']} for r in records]).to_csv(output/'capacity_choices.csv',index=False)
    print(f"Capacity decisions: {sum(x['match'] for x in records)}/{len(records)} match saved choices.")
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=Path('outputs/capacity'))
    a=p.parse_args()
    try:result=run(a.repository,a.output)
    except (OSError,ValueError) as e:p.exit(2,str(e)+'\n')
    sys.exit(0 if result['all_match'] and result['groups']==27 else 1)

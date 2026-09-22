#!/usr/bin/env python3
"""Small component-level execution of preserved research functions.

Uses 512 graph nodes and 64 disjoint query points in three synthetic test geometries.
Optional --data-root uses external catalogs instead; those catalogs are not bundled.
Closure labels are synthetic in-span fields, NOT new cosmological-validation
results. No production driver, historical cache, full sweep, or plot is run.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, runpy, sys, time, ast
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, sparse
from original_function_loader import load_functions

EXPECTED_SOURCES={
 'common/mock23_current_samples_common_single_cell_v2.py':'74a22a9c7c72bb6a73071a39109e7bf9eb1d8a6fcd2668b7f14852af1dad8fc7',
 'paper_ii/mock1_potential_flow_full_robustness_single_cell_v5.py':'1afd080b7a05aaa068374a7e1baa5f972a1ae5fbc1b1043a5f000abb9c179612',
 'paper_ii/mock2_radial_potential_highmode_capacity_h3_resume_safe_v2.py':'5af5cc6ae77f79a4056fd0adbc84d7eab259bffc170ae2c3252b680b24f777c1',
}


def synthetic_positions(case: int) -> np.ndarray:
 """Independent test geometry, not a simulation catalog or paper data."""
 rng = np.random.default_rng(260922 + case)
 n = 768
 directions = rng.normal(size=(n, 3))
 directions /= np.linalg.norm(directions, axis=1)[:, None]
 radii = 12.0 + 95.0 * rng.random(n) ** (1.0 / 3.0)
 x = directions * radii[:, None]
 if case == 2:
  x *= np.array([1.0, 0.75, 1.15])
 elif case == 3:
  x += 9.0 * np.tanh(x / 40.0)
 return x

def digest(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def relerr(a,b):
 den=float(np.linalg.norm(b))
 return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/den) if den>0 else float(np.linalg.norm(a))

def run(repo:Path,root:Path | None,output:Path):
 start=time.perf_counter();src=repo/'research_sources';output.mkdir(parents=True,exist_ok=True)
 if (output/'component_smoke.json').exists(): raise FileExistsError('Choose a fresh output directory; an audit already exists.')
 for rel,expected in EXPECTED_SOURCES.items():
  if expected and digest(src/rel)!=expected: raise ValueError('Source fingerprint mismatch: '+rel)
 # The shared file contains only imports and function definitions. Do not permit
 # an accidentally substituted unguarded production program to be executed here.
 text=(src/'common/mock23_current_samples_common_single_cell_v2.py').read_text()
 assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Expr)) for n in ast.parse(text).body)
 assert all(isinstance(n.value,ast.Constant) for n in ast.parse(text).body if isinstance(n,ast.Expr))
 common=runpy.run_path(str(src/'common/mock23_current_samples_common_single_cell_v2.py'))
 v5,e5=load_functions(src/'paper_ii/mock1_potential_flow_full_robustness_single_cell_v5.py',
   ['make_label_split','compute_centered_coordinate_metric','invert_local_metric',
    'gradients_from_markov_covariance','potential_radial_design','potential_velocity','design_spectrum'],
   ['METRIC_EIGENVALUE_RELATIVE_FLOOR','METRIC_EIGENVALUE_ABSOLUTE_FLOOR'],common)
 h3,eh=load_functions(src/'paper_ii/mock2_radial_potential_highmode_capacity_h3_resume_safe_v2.py',
   ['_build_base_kernel','_observed_markov_from_base_kernel','_centered_coordinate_metric_sparse',
    '_invert_local_metric','_observed_function_gradient_block','_query_transition','_query_metric',
    '_query_function_gradient_block','_weighted_crossproducts','_cholesky_with_jitter',
    '_solve_leading_cholesky','_regularization_diagonal','_candidate_grid_coefficients',
    '_single_velocity_prediction','_many_velocity_predictions'],
   ['METRIC_EIGENVALUE_RELATIVE_FLOOR','METRIC_EIGENVALUE_ABSOLUTE_FLOOR','N_AFFINE_MODES','NUMERICAL_RIDGE'],common)
 paths=['mock1_complete_sphere/mock1.npz','mock2_schechter_selection/mock2.npz','mock3_inhomogeneous_survey/mock3.npz']
 records=[];checks=[]
 def check(name,val,threshold=None):
  ok=bool(val) if threshold is None else bool(np.isfinite(val) and val<=threshold)
  checks.append({'name':name,'value':float(val) if threshold is not None else bool(val),'upper_bound':threshold,'passed':ok})
 for j,rel in enumerate(paths,1):
  if root is None:
   pos = synthetic_positions(j)
   ids = np.arange(len(pos), dtype=np.int64)
  else:
   with np.load(root/rel,allow_pickle=False) as z:
    pos=z['pos']-np.array([250.,250.,250.]); ids=z['ids']
  rng=np.random.default_rng(9100+j);order=rng.permutation(len(pos));ii=np.sort(order[:512]);qq=np.sort(order[512:576])
  x=pos[ii]; qx=pos[qq]; los=x/np.linalg.norm(x,axis=1)[:,None]
  check(f'case{j}_small_graph_query_ids_disjoint',not len(np.intersect1d(ids[ii],ids[qq])))
  b=common['_build_base_kernel'](x,48,16,1.)
  hb=h3['_build_base_kernel'](x,48,16,1.)
  check(f'case{j}_kernel_paths_agree',np.max(np.abs((b['kernel']-hb['kernel']).toarray())),1e-13)
  P=h3['_observed_markov_from_base_kernel'](b)
  check(f'case{j}_markov_row_sum',float(np.max(np.abs(np.asarray(P.sum(axis=1)).ravel()-1))),1e-12)
  # eigsh's arbitrary initialization can differ across SciPy versions. Small
  # numerical tolerances are checked; no bitwise eigenvector equality is claimed.
  np.random.seed(1930+j)
  geo=common['_build_weighted_geometry'](b,np.ones(len(x)),0.,33); phi=geo['eigenfunctions'];pi=geo['stationary']
  check(f'case{j}_stationarity',relerr(P.T@pi,pi),1e-12)
  check(f'case{j}_eigen_residual',relerr(P@phi,phi*geo['eigenvalues'][None,:]),1e-7)
  check(f'case{j}_weighted_orthogonality',float(np.max(np.abs(phi.T@(pi[:,None]*phi)-np.eye(33)))),1e-9)
  check(f'case{j}_constant_mode',float(np.max(np.abs(phi[:,0]-np.mean(phi[:,0])))),1e-8)
  G,mean=h3['_centered_coordinate_metric_sparse'](P,x,b['rho'])
  inv,eig,condition,clipped=h3['_invert_local_metric'](G)
  check(f'case{j}_all_metric_directions_retained',np.all(clipped==0))
  ti=h3['_query_transition'](qx,b);Gq,mq=h3['_query_metric'](ti,x)
  iq,eq,cq,clipq=h3['_invert_local_metric'](Gq)
  check(f'case{j}_query_transition_row_sum',float(np.max(np.abs(ti['transition'].sum(1)-1))),1e-12)
  check(f'case{j}_query_metric_directions_retained',np.all(clipq==0))
  beta=np.array([.7,-1.2,2.3]); f=np.column_stack([np.ones(len(x)),x,2.5+x@beta])
  expected=np.vstack([np.zeros(3),np.eye(3),beta])
  go=h3['_observed_function_gradient_block'](P,x,f,b['rho'],inv,mean)
  gq=h3['_query_function_gradient_block'](ti,x,f,iq,mq)
  check(f'case{j}_observed_affine_gradients',float(np.max(np.abs(go-expected[None,:,:]))),1e-9)
  check(f'case{j}_query_affine_gradients',float(np.max(np.abs(gq-expected[None,:,:]))),1e-9)
  G5,m5=v5['compute_centered_coordinate_metric'](P,x,b['rho']);inv5,*_=v5['invert_local_metric'](G5)
  g5=v5['gradients_from_markov_covariance'](P,x,f,b['rho'],inv5,m5,chunk_size=5)
  check(f'case{j}_two_original_gradient_paths_agree',float(np.max(np.abs(go-g5))),1e-10)
  gp=h3['_observed_function_gradient_block'](P,x,phi[:,1:],b['rho'],inv,mean)
  gqp=h3['_query_function_gradient_block'](ti,x,phi[:,1:],iq,mq)
  H=np.concatenate([np.broadcast_to(np.eye(3),(len(x),3,3)),gp],axis=1)
  Hq=np.concatenate([np.broadcast_to(np.eye(3),(len(qx),3,3)),gqp],axis=1)
  A=v5['potential_radial_design'](H,los);split=v5['make_label_split'](len(x),.7,.15,9200+j)
  train=split['train'];val=split['validation'];test=split['test']
  theta=rng.standard_normal(35);truth=v5['potential_velocity'](H,theta,35);tquery=v5['potential_velocity'](Hq,theta,35)
  y=A@theta;spec=v5['design_spectrum'](A[train]);check(f'case{j}_radial_design_full_rank',spec['numerical_rank']==35)
  # Exact in-span labels, unpenalized original Cholesky pathway.
  errors=[]
  for label,w in [('unweighted',np.ones(len(train))),('positive_weighted',np.linspace(.5,1.5,len(train)))]:
   gram,rhs=h3['_weighted_crossproducts'](A[train],y[train],w)
   L,jitter=h3['_cholesky_with_jitter'](gram);th=h3['_solve_leading_cholesky'](L,rhs,35)
   check(f'case{j}_{label}_closure_no_jitter',jitter==0.)
   errs={'coefficients':relerr(th,theta),'heldout_radial':relerr(A[test]@th,y[test]),'heldout_3d':relerr(v5['potential_velocity'](H[test],th,35),truth[test]),'query_3d':relerr(v5['potential_velocity'](Hq,th,35),tquery)}
   for k,v in errs.items():check(f'case{j}_{label}_closure_{k}',v,1e-8)
   errors.append({'weighting':label,**errs})
  # Paper I Cartesian fitting and Nystrom extension from the same preserved code.
  ext=common['_nystrom_extend'](qx,x,b,geo)['eigenfunctions']
  manual=np.einsum('ij,ijm->im',ti['transition'],phi[ti['indices']],optimize=True)/geo['eigenvalues'][None,:]
  check(f'case{j}_nystrom_uses_same_query_transition',float(np.max(np.abs(ext-manual))),1e-10)
  coeff=rng.standard_normal((33,3));v=phi@coeff;vcq=ext@coeff
  fitted=common['_fit_weighted_spectral'](phi[train],v[train],np.ones(len(train)),geo['generator_eigenvalues'],0.,0.)
  cart_errors={'coefficients':relerr(fitted,coeff),'heldout_3d':relerr(phi[test]@fitted,v[test]),'query_3d':relerr(ext@fitted,vcq)}
  for k,err in cart_errors.items():check(f'case{j}_cartesian_closure_{k}',err,1e-8)
  # Noisy radial finite-grid fit through original candidate builder. Selection
  # uses only noisy radial validation; hidden vectors never enter this choice.
  yn=y+.1*np.sqrt(np.mean(y[train]**2))*rng.standard_normal(len(x))
  gram,rhs=h3['_weighted_crossproducts'](A[train],yn[train],np.ones(len(train)))
  vals=np.r_[np.zeros(3),geo['generator_eigenvalues'][1:]]
  cc,grid=h3['_candidate_grid_coefficients'](gram,rhs,vals,[3,11,35],[0.,1e-4,1e-2],[1.,2.])
  score=np.sqrt(np.mean((A[val]@cc-yn[val,None])**2,axis=0)/np.mean(yn[val]**2));best=int(np.argmin(score))
  row=grid.iloc[best];count=int(row.n_basis)
  fit=np.sort(np.r_[train,val]);gg,rr=h3['_weighted_crossproducts'](A[fit,:count],yn[fit],np.ones(len(fit)))
  penalty=h3['_regularization_diagonal'](vals[:count],float(row.penalty_power))
  system=gg+np.diag(float(row.regularization_strength)*penalty+h3['NUMERICAL_RIDGE'])
  L,jitter=h3['_cholesky_with_jitter'](system);cf=h3['_solve_leading_cholesky'](L,rr,count)
  pred=h3['_single_velocity_prediction'](gqp,cf,count,1.)
  check(f'case{j}_noisy_radial_selection_refit_query_finite',np.all(np.isfinite(score)) and np.all(np.isfinite(pred)))
  many=h3['_many_velocity_predictions'](gqp,cc,1.)
  check(f'case{j}_two_original_velocity_assemblies_agree',float(np.max(np.abs(many[:,:,best]-v5['potential_velocity'](Hq,cc[:,best],35)))),1e-10)
  records.append({'test_case':j,'input_catalog_sha256':digest(root/rel) if root is not None else None,'graph_count':len(x),'query_count':len(qx),'graph_index_sha256':hashlib.sha256(ii.tobytes()).hexdigest(),'query_index_sha256':hashlib.sha256(qq.tobytes()).hexdigest(),
    'markov_modes':33,'potential_parameters':35,'kernel_neighbors':48,'bandwidth_neighbor':16,'generator_normalization':'median of 32 nonconstant smoke modes, original common helper; NOT paper high-mode 511-mode scale',
    'design_condition':float(spec['condition_number']),'observed_metric_max_condition':float(condition.max()),'query_metric_max_condition':float(cq.max()),
    'radial_closure_errors':errors,'cartesian_closure_errors':cart_errors,'noisy_grid_candidates':len(grid),'selected_noisy_radial_configuration':{'parameter_count':count,'lambda':float(row.regularization_strength),'power':float(row.penalty_power),'radial_validation_nrmse':float(score[best])}})
 result={'scope':'Original-function component smoke; exact in-span fields and a small synthetic-noise grid on independently generated synthetic positions by default; optional user-supplied catalog positions. NOT full-paper performance reproduction.',
 'production_main_workflows_executed':False,'original_cache_used':False,'scientific_function_bodies_modified':False,'release_source_default_paths_portabilized':True,'input_mode':'synthetic' if root is None else 'external_catalogs',
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__},
 'sources':[{'source_filename':'mock23_current_samples_common_single_cell_v2.py','sha256':digest(src/'common/mock23_current_samples_common_single_cell_v2.py'),'loading_mode':'full shared definition module, no top-level driver'},e5,eh],
 'records':records,'checks':checks,'check_count':len(checks),'all_checks_passed':all(c['passed'] for c in checks),'elapsed_seconds':time.perf_counter()-start,
 'limitation':'Component checks only. No high-mode/cache/sweep/physical-accuracy/coverage/bootstrap validation or paper figure reproduction. This run records its interpreter; installation evidence is separate.'}
 (output/'component_smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print('Smoke checks',len(checks),'passed',sum(c['passed'] for c in checks));print('Failed:',[c for c in checks if not c['passed']]);
 print('Closure errors:',json.dumps([{'test_case':r['test_case'],'radial':r['radial_closure_errors'],'cartesian':r['cartesian_closure_errors']} for r in records],indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
 p.add_argument('--data-root',type=Path,default=None,help='Optional external catalogs; omitted for the self-contained synthetic demo.')
 p.add_argument('--output',type=Path,default=Path('outputs/demo'))
 a=p.parse_args()
 try:
  result=run(a.repository.resolve(),a.data_root.resolve() if a.data_root else None,a.output)
 except (OSError,ValueError) as exc:
  p.exit(2, f'{type(exc).__name__}: {exc}\n')
 sys.exit(0 if result['all_checks_passed'] else 1)

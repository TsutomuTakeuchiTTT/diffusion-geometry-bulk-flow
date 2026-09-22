import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from recompute_capacity import choices,run as capacity_run
from export_table_review import run as export_run
from demo import synthetic_positions
from verify_release import run as verify_run

class ReleaseTests(unittest.TestCase):
    def test_synthetic_geometry(self):
        for j in [1,2,3]:
            x=synthetic_positions(j);self.assertEqual(x.shape,(768,3));self.assertTrue(np.all(np.isfinite(x)))
            self.assertEqual(len(np.unique(x,axis=0)),768)
            self.assertTrue(np.all(np.linalg.norm(x,axis=1)>0))
    def test_deterministic_geometry(self):
        np.testing.assert_array_equal(synthetic_positions(2),synthetic_positions(2))
    def test_one_se_not_restricted_by_plateau_floor(self):
        out=choices([3,387,514],[.8,.51,.5],[.02,.02,.02],lambda b,m:514)
        self.assertEqual(out['one_standard_error_basis'],387)
        self.assertEqual(out['combined_practical_basis'],514)
    def test_missing_plateau(self):
        with self.assertRaises(ValueError):choices([3,11,35],[3,2,1],[0,0,0],lambda b,m:None)
    def test_duplicate_capacity_rejected(self):
        with self.assertRaises(ValueError):choices([3,3,35],[3,2,1],[0,0,0],lambda b,m:3)
    def test_nonfinite_risk_rejected(self):
        with self.assertRaises(ValueError):choices([3,11,35],[3,np.nan,1],[0,0,0],lambda b,m:3)
    def test_capacity_all_27(self):
        with tempfile.TemporaryDirectory() as d:
            o=Path(d)/'capacity';r=capacity_run(ROOT,o)
            self.assertEqual(r['groups'],27);self.assertTrue(r['all_match'])
            with self.assertRaises(FileExistsError):capacity_run(ROOT,o)
    def test_export_preserves_known_discrepancies(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'review.csv';rows=export_run(ROOT,out)
            self.assertEqual(len(rows),373);self.assertEqual(sum(not r['within_printed_rounding'] for r in rows),2)
            with self.assertRaises(FileExistsError):export_run(ROOT,out)
    def test_no_raw_catalogs_or_fonts(self):
        files=[p for p in ROOT.rglob('*') if p.is_file() and 'outputs' not in p.parts]
        self.assertFalse(any(p.suffix in {'.npz','.npy','.ttf','.otf'} for p in files))
    def test_no_private_user_paths(self):
        for p in ROOT.rglob('*'):
            if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.txt','.cff'} and 'outputs' not in p.parts:
                text=p.read_text(errors='replace')
                self.assertNotIn('C:'+chr(92)+'Users'+chr(92),text,str(p))
                self.assertNotIn('C:'+chr(92)*2+'Users'+chr(92)*2,text,str(p))
    def test_source_count_and_path_scope(self):
        m=json.loads((ROOT/'metadata/source_manifest.json').read_text())
        self.assertEqual(len(m['sources']),22)
        self.assertTrue(all(r['all_other_ast_unchanged'] for r in m['sources']))
    def test_citation_yaml(self):
        import yaml
        c=yaml.safe_load((ROOT/'CITATION.cff').read_text())
        self.assertEqual(c['cff-version'],'1.2.0');self.assertEqual(len(c['authors']),4)
        state=json.loads((ROOT/'metadata/registration_state.json').read_text())
        if state['version_doi']:
            self.assertEqual(c['doi'],state['version_doi']);self.assertEqual(c['repository-code'],state['repository_url'])
        else:
            self.assertNotIn('doi',c);self.assertNotIn('repository-code',c)

if __name__=='__main__':unittest.main()

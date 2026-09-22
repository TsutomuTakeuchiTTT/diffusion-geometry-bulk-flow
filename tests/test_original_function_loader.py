"""Adapter tests only; numerical checks are a separately executed command."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from original_function_loader import load_functions

class OriginalFunctionLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'source.py'
    def tearDown(self):self.tmp.cleanup()
    def test_original_body_and_literal_constant(self):
        self.path.write_text('FACTOR=2\ndef f(x):\n    return FACTOR*x\n')
        ns,e=load_functions(self.path,['f'],['FACTOR'],{})
        self.assertEqual(ns['f'](3),6)
        self.assertEqual(e['literal_constants'],{'FACTOR':2})
    def test_unguarded_driver_not_executed(self):
        self.path.write_text("raise RuntimeError('production must not run')\ndef f():\n    return 7\n")
        ns,e=load_functions(self.path,['f'],[],{})
        self.assertEqual(ns['f'](),7)
    def test_missing_function_rejected(self):
        self.path.write_text('def f():\n    return 0\n')
        with self.assertRaises(ValueError):load_functions(self.path,['other'],[],{})
    def test_nonliteral_configuration_not_executed(self):
        self.path.write_text("FACTOR=__import__('os').system('never executed')\ndef f():\n    return FACTOR\n")
        with self.assertRaises(ValueError):load_functions(self.path,['f'],['FACTOR'],{})
    def test_input_source_not_changed(self):
        self.path.write_text('def f(x):\n    return x*x\n')
        raw=self.path.read_bytes();mtime=self.path.stat().st_mtime_ns
        ns,e=load_functions(self.path,['f'],[],{});ns['f'](4)
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.path.stat().st_mtime_ns,mtime)
    def test_function_line_and_hash_recorded(self):
        self.path.write_text('C=2\n\ndef f(x):\n    return x+C\n')
        ns,e=load_functions(self.path,['f'],['C'],{})
        self.assertEqual(e['functions'][0]['start_line'],3)
        self.assertEqual(e['functions'][0]['end_line'],4)
        self.assertEqual(e['source_sha256'],hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(len(e['functions'][0]['body_sha256']),64)
    def test_future_annotations_do_not_require_unknown_types(self):
        self.path.write_text('def f(x: UnloadedType):\n    return x\n')
        ns,e=load_functions(self.path,['f'],[],{})
        self.assertEqual(ns['f'](4),4)

if __name__=='__main__':unittest.main()

"""Load selected unmodified functions, never an unguarded production driver.

This is a component-test adapter, NOT execution of the original main workflow.
Only explicitly named top-level function definitions and literal constants are
compiled. Function bodies are retained byte-for-byte in the source, and their
line ranges/digests are reported. The caller supplies the scientific imports.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
from typing import Any

def load_functions(path: Path, names: list[str], constants: list[str], namespace: dict[str, Any]):
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text, filename=path.name)
    fn = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    literal = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    literal[target.id] = ast.literal_eval(node.value)
    missing = set(names)-set(fn) | (set(constants)-set(literal))
    if missing: raise ValueError(f'Missing explicit definitions: {sorted(missing)}')
    ns = dict(namespace)
    ns.update(literal)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.Module(body=[future]+[fn[n] for n in names],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path.name, 'exec'),ns)
    evidence = {'source_filename':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'loading_mode':'selected_original_functions_only; production top-level driver NOT executed',
                'literal_constants':literal,'functions':[]}
    for name in names:
        n = fn[name]; segment=ast.get_source_segment(text,n)
        evidence['functions'].append({'name':name,'start_line':n.lineno,'end_line':n.end_lineno,
                                     'body_sha256':hashlib.sha256(segment.encode()).hexdigest()})
    return ns,evidence

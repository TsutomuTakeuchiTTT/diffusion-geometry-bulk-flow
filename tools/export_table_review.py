#!/usr/bin/env python3
"""Export the archived 373-cell paper/CSV comparison as a readable table.

This serializes the previous comparison record; it does not recompute regressions,
re-identify rows in source tables, or resolve bootstrap intervals anew.
"""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(repo,output):
    if output.exists():raise FileExistsError('Use a new output filename.')
    report=json.loads((repo/'metadata/numeric_table_audit.json').read_text())
    rows=[]
    for row in report['checks']:
        sources=row['source'] if isinstance(row['source'],list) else [row['source']]
        for rel in sources:
            if not (repo/'results/saved'/rel).is_file():raise FileNotFoundError(rel)
        rows.append({'paper':row['paper'],'table':row['table'],'cell':row['locator'],
           'manuscript_value':row['printed'],'saved_value':row['saved'],
           'within_printed_rounding':row['pass'],'source':'; '.join(sources),'source_field':row['source_field']})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Exported {len(rows)} archived cell checks; {sum(not x["within_printed_rounding"] for x in rows)} documented rounding differences retained.')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=Path('outputs/paper_table_review.csv'));a=p.parse_args()
    try:run(a.repository,a.output)
    except (OSError,ValueError) as e:p.exit(2,str(e)+'\n')

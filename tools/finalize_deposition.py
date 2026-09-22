#!/usr/bin/env python3
"""Bind REAL, maintainer-supplied Zenodo/GitHub identifiers in a NEW local copy.

No login, network request, upload, Git commit or remote publication is performed.
A syntactically valid DOI is not proof of reservation, ownership or publication.
Use the version DOI reserved for this exact software record, never a paper DOI.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'metadata/release_checksums.json'
SKIP = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints', '.pytest_cache', 'outputs'}
DOI_RE = re.compile(r'10\.5281/zenodo\.[1-9][0-9]*')
EXPECTED_ORCIDS = ['0000-0001-8416-7673', '0009-0001-5284-6759', '0000-0002-5237-9433', '0009-0001-9463-0673']
BEGIN = '<!-- RELEASE-IDENTIFIERS:BEGIN -->'
END = '<!-- RELEASE-IDENTIFIERS:END -->'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def yaml_text(value: dict) -> str:
    """Emit a deterministic YAML subset using JSON-quoted scalar values."""
    def scalar(x):
        if x is None:
            return 'null'
        if not isinstance(x, (str, bool, int, float)):
            raise TypeError('Unexpected YAML scalar')
        return json.dumps(x, ensure_ascii=False, allow_nan=False)
    def lines(obj, indent=0):
        pad = ' ' * indent
        if isinstance(obj, dict):
            for key, val in obj.items():
                if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', key):
                    raise ValueError('Unsupported metadata key')
                if isinstance(val, (dict, list)):
                    yield f'{pad}{key}:'
                    yield from lines(val, indent + 2)
                else:
                    yield f'{pad}{key}: {scalar(val)}'
        elif isinstance(obj, list):
            for val in obj:
                if isinstance(val, (dict, list)):
                    yield pad + '-'
                    yield from lines(val, indent + 2)
                else:
                    yield pad + '- ' + scalar(val)
        else:
            raise TypeError('Root YAML values must be structured')
    return '\n'.join(lines(value)) + '\n'


def normalize_doi(raw: str) -> str:
    value = raw.strip()
    if value.lower().startswith('doi:'):
        value = value[4:].strip()
    if value.lower().startswith('https://doi.org/'):
        value = value[len('https://doi.org/'):]
    if not DOI_RE.fullmatch(value):
        raise ValueError('Use the actual version DOI reserved in Zenodo: 10.5281/zenodo.<digits>.')
    return value


def normalize_repository(raw: str) -> str:
    value = raw.strip().rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or parsed.netloc.lower() != 'github.com' or parsed.query or parsed.fragment:
        raise ValueError('Use an actual HTTPS GitHub repository URL without credentials, query or fragment.')
    parts = parsed.path.strip('/').split('/')
    if len(parts) != 2 or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?', parts[0]):
        raise ValueError('The GitHub URL must identify exactly one owner/repository.')
    name = parts[1]
    if name.endswith('.git'):
        name = name[:-4]
    if not name or name in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('Invalid GitHub repository name.')
    return f'https://github.com/{parts[0]}/{name}'


def normalize_date(raw: str) -> str:
    value = raw.strip()
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('Use a release date in YYYY-MM-DD form.')
    if parsed.year < 2000:
        raise ValueError('Unexpected release year.')
    return value


def orcid_valid(value: str) -> bool:
    if not re.fullmatch(r'[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]', value):
        return False
    compact = value.replace('-', '')
    total = 0
    for ch in compact[:15]:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    return compact[-1] == ('X' if check == 10 else str(check))


def distribution_files(repo: Path) -> list[Path]:
    return sorted(p for p in repo.rglob('*') if p.is_file() and not any(s in SKIP for s in p.relative_to(repo).parts))


def verify_input_files(repo: Path) -> dict:
    data = read_json(repo / MANIFEST)
    allowed = {MANIFEST}
    errors = []
    for row in data['files']:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in row['path'] or row['path'] in allowed:
            raise ValueError('Unsafe or duplicated manifest path.')
        allowed.add(row['path'])
        path = repo / rel.as_posix()
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo.resolve()):
            errors.append('Missing or unsafe file: ' + row['path'])
        else:
            content = path.read_bytes()
            if len(content) != row['size_bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
                errors.append('Changed file: ' + row['path'])
    errors.extend('Unlisted file: ' + p.relative_to(repo).as_posix() for p in distribution_files(repo)
                  if p.relative_to(repo).as_posix() not in allowed)
    if errors:
        raise ValueError('Distribution check failed: ' + '; '.join(errors[:8]))
    return {'verified_files': len(data['files']), 'passed': True}


def refresh_checksums(repo: Path) -> None:
    rows = []
    for p in distribution_files(repo):
        if p == repo / MANIFEST:
            continue
        content = p.read_bytes()
        rows.append({'path': p.relative_to(repo).as_posix(), 'size_bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    write_json(repo / MANIFEST, {'scope': 'Distributed files excluding this self-referential manifest and generated/local directories.', 'files': rows})


def check_metadata(repo: Path, *, require_identifiers: bool = False) -> dict:
    authors = read_json(repo / 'metadata/software_authors.json')['authors']
    state = read_json(repo / 'metadata/registration_state.json')
    form = read_json(repo / 'metadata/zenodo_registration.json')
    cff = read_json(repo / 'metadata/citation_metadata.json')
    if [a['orcid'] for a in authors] != EXPECTED_ORCIDS or not all(orcid_valid(a['orcid']) for a in authors):
        raise ValueError('Confirmed author order or ORCID checksum mismatch.')
    if len(cff['authors']) != 4 or len(form['creators']) != 4:
        raise ValueError('Exactly four confirmed software creators are required.')
    for a, c, z in zip(authors, cff['authors'], form['creators']):
        if (c['family-names'], c['given-names'], c['orcid'], c['affiliation']) != (a['family_name'], a['given_names'], 'https://orcid.org/'+a['orcid'], '; '.join(a['affiliations'])):
            raise ValueError('CFF author mismatch.')
        if (z['name'], z['orcid'], z['affiliations']) != (a['family_name']+', '+a['given_names'], a['orcid'], a['affiliations']):
            raise ValueError('Zenodo author mismatch.')
    if not state['authors_confirmed'] or state['software_version'] != (repo / 'VERSION').read_text().strip():
        raise ValueError('Release state/version mismatch.')
    if state['github_tag'] != 'v'+state['software_version'] or cff['version'] != state['software_version'] or form['version'] != state['software_version']:
        raise ValueError('Version/tag metadata mismatch.')
    if (repo / 'CITATION.cff').read_text(encoding='utf-8') != yaml_text(cff):
        raise ValueError('CITATION.cff differs from canonical citation metadata.')
    if cff['license'] != 'BSD-3-Clause' or [l['spdx'] for l in form['licenses']] != ['BSD-3-Clause', 'CC-BY-4.0']:
        raise ValueError('Component license metadata mismatch.')
    missing = [k for k in ['version_doi', 'repository_url', 'publication_date'] if not state[k]]
    if missing:
        if any(state[k] for k in ['version_doi', 'repository_url', 'publication_date']):
            raise ValueError('Partially bound identifiers are not permitted.')
        if any(k in cff for k in ['doi', 'repository-code', 'date-released']):
            raise ValueError('Unexpected identifier in unbound CFF.')
        if require_identifiers:
            raise ValueError('Still required from the real accounts: ' + ', '.join(missing))
    else:
        doi = normalize_doi(state['version_doi'])
        url = normalize_repository(state['repository_url'])
        released = normalize_date(state['publication_date'])
        if (cff['doi'], cff['repository-code'], cff['date-released']) != (doi, url, released):
            raise ValueError('Bound CFF identifiers mismatch.')
        if (form['version_doi'], form['repository_url'], form['publication_date']) != (doi, url, released):
            raise ValueError('Bound Zenodo identifiers mismatch.')
        if form['github_release_url'] != url+'/releases/tag/'+state['github_tag']:
            raise ValueError('GitHub release URL mismatch.')
    return {'version': state['software_version'], 'confirmed_authors': 4, 'author_order_and_orcids_match': True,
            'metadata_consistent': True, 'identifiers_bound': not missing, 'missing_real_account_fields': missing,
            'remote_publication_checked': False, 'note': 'Local metadata checks never establish DOI ownership or publication.'}


def worksheet_text(form: dict) -> str:
    out = ['Zenodo software registration: '+form['title'], 'Version: '+form['version'],
           'Resource type: Software', 'Visibility: Public', 'Publisher: Zenodo',
           'Publication date: '+(form['publication_date'] or '(actual publication date required)'),
           'Version DOI: '+(form['version_doi'] or '(reserve in Zenodo; not yet assigned here)'),
           'Repository: '+(form['repository_url'] or '(actual GitHub repository URL required)'),
           '\nCREATORS (keep this order)']
    for i, a in enumerate(form['creators'], 1):
        out.extend([f"{i}. {a['name']}", 'ORCID: '+a['orcid'], *['Affiliation: '+s for s in a['affiliations']]])
    out.extend(['\nDESCRIPTION', '\n\n'.join(form['description_paragraphs']), '\nLICENSES (different file scopes)'])
    for lic in form['licenses']:
        out.append(lic['spdx']+': '+lic['applies_to'])
    out.extend(['\nKEYWORDS', '; '.join(form['keywords']), '\nRELATED IDENTIFIERS'])
    for x in form['related_identifiers']:
        out.append(x['relation']+': '+x['identifier'])
    out.append('\nDo not enable a second automatic GitHub archive for this same version. Do not use a paper DOI.')
    return '\n'.join(out)+'\n'


def render_publication_templates(repo: Path, *, bound: bool) -> None:
    form = read_json(repo / 'metadata/zenodo_registration.json')
    (repo / 'docs/zenodo_registration_fields.txt').write_text(worksheet_text(form), encoding='utf-8', newline='\n')
    if not bound:
        return
    values = {'ZENODO_VERSION_DOI': form['version_doi'], 'GITHUB_REPOSITORY_URL': form['repository_url'],
              'RELEASE_YEAR': form['publication_date'][:4], 'RELEASE_VERSION': form['version']}
    for template in sorted((repo / 'publication/templates').glob('*.in')):
        content = template.read_text(encoding='utf-8')
        for key, val in values.items():
            content = content.replace('@'+key+'@', val)
        if re.search(r'@[A-Z_]+@', content):
            raise ValueError('Unresolved template token.')
        (repo / 'publication' / template.name[:-3]).write_text(content, encoding='utf-8', newline='\n')


def make_archive(repo: Path, path: Path, released: str) -> None:
    dt = date.fromisoformat(released)
    if path.exists():
        raise FileExistsError(path)
    with ZipFile(path, 'x', compression=ZIP_DEFLATED, compresslevel=9) as z:
        for p in distribution_files(repo):
            if p.is_symlink():
                raise ValueError('Symlink not permitted in publication archive.')
            info = ZipInfo('diffusion-geometry-bulk-flow/'+p.relative_to(repo).as_posix(), (dt.year,dt.month,dt.day,0,0,0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes(), compresslevel=9)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix+'.sha256').write_text(checksum+'  '+path.name+'\n', encoding='ascii', newline='\n')


def finalize(repo: Path, destination: Path, *, doi: str, repository_url: str, publication_date: str, confirmed: bool) -> dict:
    if not confirmed:
        raise ValueError('Confirm that the supplied DOI is the actual VERSION DOI reserved for this record.')
    doi = normalize_doi(doi); repository_url = normalize_repository(repository_url); publication_date = normalize_date(publication_date)
    repo = repo.resolve(); destination = destination.absolute()
    if destination.exists():
        raise FileExistsError('Output already exists; choose a new directory.')
    if destination.resolve().is_relative_to(repo):
        raise ValueError('Output must be outside the input release folder.')
    verify_input_files(repo)
    check_metadata(repo)
    source_state = read_json(repo / 'metadata/registration_state.json')
    if source_state['version_doi']:
        raise ValueError('This copy is already bound. Use the original unbound registration-ready package.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dg-finalize-', dir=destination.parent) as temp:
        work = Path(temp); target = work/'diffusion-geometry-bulk-flow'
        for source in distribution_files(repo):
            new = target/source.relative_to(repo); new.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,new)
        state = read_json(target/'metadata/registration_state.json')
        state.update(version_doi=doi, repository_url=repository_url, publication_date=publication_date,
                     identifier_binding='maintainer_supplied_reserved_version_doi', external_publication_verified=False)
        write_json(target/'metadata/registration_state.json',state)
        cff = read_json(target/'metadata/citation_metadata.json')
        cff.update(doi=doi, **{'repository-code':repository_url, 'url':'https://doi.org/'+doi, 'date-released':publication_date})
        write_json(target/'metadata/citation_metadata.json',cff)
        (target/'CITATION.cff').write_text(yaml_text(cff),encoding='utf-8',newline='\n')
        form = read_json(target/'metadata/zenodo_registration.json')
        release_url = repository_url+'/releases/tag/'+state['github_tag']
        form.update(version_doi=doi, repository_url=repository_url, publication_date=publication_date, github_release_url=release_url)
        form['related_identifiers']=[{'relation':'isIdenticalTo', 'identifier':release_url, 'scheme':'url', 'resource_type':'software'}]
        write_json(target/'metadata/zenodo_registration.json',form)
        paper = read_json(target/'metadata/papers.json');paper.update(doi=doi,repository_url=repository_url)
        write_json(target/'metadata/papers.json',paper)
        readme=(target/'README.md').read_text(encoding='utf-8')
        block=f'{BEGIN}\nSoftware version DOI: https://doi.org/{doi}\n\nRepository: {repository_url}\n\nRelease: {release_url}\n\nThe Zenodo record, not the presence of this metadata, establishes public availability.\n{END}'
        if readme.count(BEGIN)!=1 or readme.count(END)!=1:raise ValueError('README identifier markers changed.')
        readme=re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END),lambda _:block,readme,flags=re.S)
        (target/'README.md').write_text(readme,encoding='utf-8',newline='\n')
        render_publication_templates(target,bound=True)
        bound_check=check_metadata(target,require_identifiers=True)
        changed=[]
        for p in distribution_files(repo):
            rel=p.relative_to(repo)
            if p.read_bytes()!=(target/rel).read_bytes():changed.append(rel.as_posix())
        immutable=[p for area in ['research_sources','results/saved'] for p in (repo/area).rglob('*') if p.is_file()]
        if not all(p.read_bytes()==(target/p.relative_to(repo)).read_bytes() for p in immutable):
            raise ValueError('Scientific source/result content changed during binding.')
        audit={'version':state['software_version'],'identifiers_bound':True,'version_doi':doi,'repository_url':repository_url,
               'publication_date':publication_date,'science_sources_and_saved_results_unchanged':True,
               'input_files_verified_unchanged':len(immutable),'changed_metadata_files':changed,
               'remote_actions_performed':False,'remote_publication_verified':False,
               'input_manifest_sha256':hashlib.sha256((repo/MANIFEST).read_bytes()).hexdigest(),
               'metadata_check':bound_check}
        write_json(target/'reports/identifier_binding_audit.json',audit)
        refresh_checksums(target);verify_input_files(target)
        archive=work/('diffusion-geometry-bulk-flow-v'+state['software_version']+'.zip')
        make_archive(target,archive,publication_date)
        write_json(work/'binding_receipt.json',audit)
        shutil.copy2(target/'docs/zenodo_registration_fields.txt',work/'zenodo_registration_fields.txt')
        work.rename(destination)
    return {'output_directory':str(destination),'archive':str(destination/archive.name),'sha256_file':str(destination/(archive.name+'.sha256')),
            'version_doi':doi,'created_locally_only':True,'published':False}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=ROOT)
    parser.add_argument('--check',action='store_true',help='Check local files and metadata only.')
    parser.add_argument('--require-identifiers',action='store_true')
    parser.add_argument('--doi');parser.add_argument('--repository-url');parser.add_argument('--publication-date')
    parser.add_argument('--confirm-reserved-version-doi',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.check:
            verify_input_files(args.repository)
            result=check_metadata(args.repository,require_identifiers=args.require_identifiers)
        else:
            if not all([args.doi,args.repository_url,args.publication_date,args.output_dir]):
                parser.error('Binding requires --doi, --repository-url, --publication-date and --output-dir.')
            result=finalize(args.repository,args.output_dir,doi=args.doi,repository_url=args.repository_url,
                            publication_date=args.publication_date,confirmed=args.confirm_reserved_version_doi)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f'Metadata operation failed: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())

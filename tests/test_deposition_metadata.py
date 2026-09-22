"""Offline metadata checks. Test identifiers below are synthetic and never published."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from finalize_deposition import (check_metadata, finalize, normalize_date, normalize_doi,
                                 normalize_repository, orcid_valid, read_json, yaml_text,
                                 verify_input_files, make_archive)

# Deliberately artificial identifiers used only inside temporary test directories.
TEST_DOI = '10.5281/zenodo.999999999999999999'
TEST_REPO = 'https://github.com/offline-fixture/diffusion-geometry-bulk-flow'

class DepositionMetadataTests(unittest.TestCase):
    def test_current_metadata(self):
        result = check_metadata(ROOT)
        self.assertEqual(result['version'], '0.1.0')
        self.assertEqual(result['confirmed_authors'], 4)
        self.assertFalse(result['remote_publication_checked'])

    def test_cff_is_yaml_and_matches_canonical(self):
        import yaml
        expected = read_json(ROOT / 'metadata/citation_metadata.json')
        actual = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(actual['license'], 'BSD-3-Clause')

    def test_doi_normalization(self):
        self.assertEqual(normalize_doi(' https://doi.org/' + TEST_DOI + ' '), TEST_DOI)
        self.assertEqual(normalize_doi('doi: ' + TEST_DOI), TEST_DOI)

    def test_paper_or_placeholder_doi_rejected(self):
        for value in ['10.1103/PhysRevD.example', 'ZENODO_VERSION_DOI', '10.5281/zenodo.123?token=x', TEST_DOI+'\nextra']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_doi(value)

    def test_repository_normalization(self):
        self.assertEqual(normalize_repository(TEST_REPO+'.git/'), TEST_REPO)

    def test_credentials_and_wrong_url_rejected(self):
        for value in ['http://github.com/a/b', 'https://user:secret@github.com/a/b',
                      TEST_REPO+'/releases/tag/v0.1.0', TEST_REPO+'?token=example',
                      'https://github.com.evil.invalid/a/b', 'https://github.com/a/..']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_repository(value)

    def test_release_date(self):
        self.assertEqual(normalize_date('2026-09-22'), '2026-09-22')
        for value in ['2026-02-30', '20260922', '2026-9-22']:
            with self.assertRaises(ValueError):
                normalize_date(value)

    def test_all_confirmed_orcid_checksums(self):
        for a in read_json(ROOT/'metadata/software_authors.json')['authors']:
            self.assertTrue(orcid_valid(a['orcid']))
        self.assertFalse(orcid_valid('0000-0001-8416-7674'))

    def test_affiliations(self):
        authors = read_json(ROOT/'metadata/software_authors.json')['authors']
        self.assertEqual(len(authors[0]['affiliations']), 2)
        self.assertTrue(all(len(a['affiliations']) == 1 for a in authors[1:]))

    def test_license_scopes_not_alternative(self):
        form = read_json(ROOT/'metadata/zenodo_registration.json')
        self.assertEqual([x['spdx'] for x in form['licenses']], ['BSD-3-Clause','CC-BY-4.0'])
        self.assertIn('not alternative licenses', '\n'.join(form['description_paragraphs']))
        self.assertEqual(form['resource_type'], 'Software')

    def test_explicit_confirmation_required(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            finalize(ROOT, Path(d)/'out', doi=TEST_DOI, repository_url=TEST_REPO,
                     publication_date='2026-09-22', confirmed=False)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileExistsError):
            finalize(ROOT, Path(d), doi=TEST_DOI, repository_url=TEST_REPO,
                     publication_date='2026-09-22', confirmed=True)

    def test_no_nested_destination(self):
        with self.assertRaises(ValueError):
            finalize(ROOT, ROOT/'forbidden_binding_output', doi=TEST_DOI, repository_url=TEST_REPO,
                     publication_date='2026-09-22', confirmed=True)

    def test_full_offline_binding_and_archive(self):
        if read_json(ROOT/'metadata/registration_state.json')['version_doi']:
            self.skipTest('Already bound release; the original unbound package is required for this fixture.')
        original_manifest = (ROOT/'metadata/release_checksums.json').read_bytes()
        with tempfile.TemporaryDirectory(prefix='dg test binding ') as d:
            target=Path(d)/'new delivery'
            result=finalize(ROOT,target,doi=TEST_DOI,repository_url=TEST_REPO,
                            publication_date='2026-09-22',confirmed=True)
            self.assertFalse(result['published'])
            out=target/'diffusion-geometry-bulk-flow'
            self.assertTrue(check_metadata(out,require_identifiers=True)['identifiers_bound'])
            self.assertTrue(verify_input_files(out)['passed'])
            for rel in ['publication/software.bib','publication/availability_PaperI.tex','publication/availability_PaperII.tex']:
                text=(out/rel).read_text()
                self.assertNotIn('@RELEASE_',text)
                self.assertNotIn('@ZENODO_',text)
            for area in ['research_sources','results/saved']:
                for p in (ROOT/area).rglob('*'):
                    if p.is_file():self.assertEqual(p.read_bytes(),(out/p.relative_to(ROOT)).read_bytes())
            self.assertEqual((ROOT/'metadata/release_checksums.json').read_bytes(),original_manifest)
            archive=Path(result['archive'])
            digest=hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(archive.with_suffix('.zip.sha256').read_text().split()[0],digest)
            with ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertTrue(all(n.startswith('diffusion-geometry-bulk-flow/') for n in z.namelist()))
                for n in z.namelist():
                    self.assertEqual(z.read(n),(out/Path(n).relative_to('diffusion-geometry-bulk-flow')).read_bytes())
            second=target/'second.zip';make_archive(out,second,'2026-09-22')
            self.assertEqual(second.read_bytes(),archive.read_bytes())

    def test_doi_and_repository_not_guessed(self):
        state=read_json(ROOT/'metadata/registration_state.json')
        if not state['version_doi']:
            with self.assertRaises(ValueError):check_metadata(ROOT,require_identifiers=True)
            self.assertIsNone(read_json(ROOT/'metadata/zenodo_registration.json')['version_doi'])

    def test_no_automatic_deposit_workflow(self):
        self.assertFalse((ROOT/'.zenodo.json').exists())
        state=read_json(ROOT/'metadata/registration_state.json')
        self.assertFalse(state['automatic_github_archiving_for_this_version'])
        self.assertFalse(state['zenodo_deposit_performed_by_builder'])

if __name__ == '__main__':
    unittest.main()

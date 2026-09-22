# Single-record release procedure: version 0.1.0

The four software authors and their order are confirmed. Code/documentation are BSD-3-Clause; saved numerical records and their source-provenance metadata are CC BY 4.0. The scientific sources and stored numerical values are frozen. Publication identifiers are supplied by the maintainer's real accounts, not generated offline.

## Chosen route: reserve the DOI, then manually deposit the exact release ZIP

1. In the production Zenodo account, create one new **Software** draft for this joint Papers I/II code release. Use the title, four creators, description, both component-specific licenses and version in `zenodo_registration_fields.txt`. Select 'No' for an existing DOI and use 'Get a DOI now!' to reserve a DOI for this draft. Save the draft and do not delete it. This DOI is not publicly registered until the record is published.
2. Identify or create the actual GitHub repository `diffusion-geometry-bulk-flow`. Do **not** enable GitHub-to-Zenodo automatic release archiving for this version. The present manual record is the single archival record; a second automatically created record would defeat that choice.
3. Bind the reserved **version DOI**, actual GitHub URL and release/publication date into a new local copy. The assistant can do this from the identifiers; the optional local tool is shown below. A syntactically accepted DOI is not proof of reservation or account ownership. No token is needed by this local tool.
4. Upload/commit the contents of the generated `diffusion-geometry-bulk-flow/` folder to the repository, preserving dotfiles. Do not commit the outer delivery folder, input catalogs, `outputs/` or old private review packages. Run the included read-only Actions verification. Its clean-runner install is a pending remote test until the run really succeeds. Resolve a failed verification before freezing the paper-associated release tag.
5. Create a GitHub draft release with tag `v0.1.0`, targeting the exact final verified commit. Copy `RELEASE_NOTES.md`, attach the generated `diffusion-geometry-bulk-flow-v0.1.0.zip` and its `.sha256`, then publish that release. Do not call the package an end-to-end survey product merely because it is a release. Do not move/reuse the tag after publication.
6. In the SAME reserved Zenodo draft, upload the exact attached GitHub release ZIP (not the registration-ready ZIP or an older rc/stage package). Enter both license scopes and the exact release URL as a related identifier ('is identical to'). The described software tree should match the release tag. Save, preview, verify all four creators/order/ORCIDs and version, then publish. Keep the DOI already reserved by this draft; do not create a new record or paste a paper DOI into its DOI field.
7. Verify that the published Zenodo record is public, displays the expected version DOI and author order, and its downloadable ZIP matches the checksum. Verify the GitHub release asset as well. Only then insert the generated availability paragraph and BibTeX entry in both manuscripts and submit.

This ordering permits the DOI to be inside the archived files without creating a duplicate software DOI. The `.zenodo.json` auto-ingestion file is intentionally absent: this release uses manual deposition and its mixed-license fields are provided by the worksheet, not silently imported into Zenodo.

## Optional offline identifier binding

From the registration-ready repository, use Python 3.11 or later. The three values below must be supplied from the real accounts and planned publication date. The command does not contact either service.

```sh
python tools/finalize_deposition.py --doi "ACTUAL_RESERVED_VERSION_DOI" --repository-url "ACTUAL_GITHUB_REPOSITORY_URL" --publication-date "YYYY-MM-DD" --confirm-reserved-version-doi --output-dir "../final_delivery_0.1.0"
```

The destination must be new and outside the input repository. No existing file is overwritten. The tool verifies the original manifest, binds citation/registration/manuscript-template fields, verifies that scientific sources and saved results are byte-identical, refreshes checksums and produces the version ZIP. A binding receipt is supplied beside it. It is still a local archive, not an upload or publication.

Check a folder without modifying it:

```sh
python tools/finalize_deposition.py --check
```

On a bound folder, `--require-identifiers` additionally rejects missing DOI/URL/date. Confirm the actual record in Zenodo separately; local validation cannot prove publication.

## Files and version discipline

The same final ZIP is the GitHub release asset and the Zenodo upload. GitHub's automatically generated 'Source code (zip)' may differ in ZIP container metadata even when its source tree matches; compare extracted file hashes, or use the explicitly attached ZIP for exact archive comparison. The SHA-256 file records the ZIP bytes; the internal manifest records individual distribution files.

If a correction is needed before publication, rebuild the final copy and recheck it. If code/results change after the release is public, create a new version/tag instead of silently replacing version 0.1.0. Use Zenodo's New version operation for the existing record, not an unrelated new upload. No registration action in this document is claimed to have already occurred.

## Official workflow references

- DOI reservation: https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- New upload and publication: https://help.zenodo.org/docs/deposit/create-new-upload/
- Creators: https://help.zenodo.org/docs/deposit/describe-records/creators/
- Mixed-license uploads: https://help.zenodo.org/docs/deposit/describe-records/licenses/
- Versioning: https://help.zenodo.org/docs/deposit/manage-versions/
- GitHub releases: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository

Workflow checked 2026-09-22. The actual services may change their form layout; the required content and the single-record rule remain the guide.

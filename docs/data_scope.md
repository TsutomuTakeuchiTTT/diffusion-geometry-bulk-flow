# Data distributed with the release

`results/saved/` contains 174 small CSV/JSON/TXT records: the reviewed 171 output/configuration/description files plus three legacy closure tables. No original cosmological catalog arrays are included. Twenty-seven source records contained private Windows path prefixes; these text prefixes are replaced by `<MOCK_DATA_ROOT>`. All numeric tokens and cells are retained. The original and distributed SHA-256 values are in `metadata/result_manifest.json`.

The sanitized JSON files are evidence and reading material, **not resumable historical cache metadata**. Their redacted path tokens are not executable input locations. Do not point a production checkpoint loader at them.

The default synthetic demonstration generates independent point configurations, not a mock galaxy catalog from the N-body simulation. Its data are generated at runtime and are never substituted for the paper input.

## External cosmological inputs

The original files are `mock1_complete_sphere/mock1.npz`, `mock2_schechter_selection/mock2.npz` and `mock3_inhomogeneous_survey/mock3.npz`. Their schema/checksums and upstream simulation description are recorded in `metadata/catalog_schema_and_provenance.json` and the saved catalog-description TXT files. An authorized user can point the demo at a local copy with `--data-root`. Obtaining permission and access to those files is separate from this code release. No claim is made that the simulation snapshot or these catalogs are publicly downloadable from this repository.

Large geometry/gradient caches, historical partition arrays, prediction bundles, full candidate grids, manuscript PDFs and fonts are omitted. A future separately archived dataset can add them after redistribution terms are settled, without changing what the present release claims to contain.

Git attributes disable line-ending conversion for `results/saved/**` so that archived CRLF records retain their distribution hashes on Windows and Linux. All other text files use LF.

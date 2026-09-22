# Diffusion geometry for galaxy bulk-flow reconstruction

**Paper-associated research-software version: 0.1.0.**

<!-- RELEASE-IDENTIFIERS:BEGIN -->
Software version DOI: https://doi.org/10.5281/zenodo.22898864

Repository: https://github.com/TsutomuTakeuchiTTT/diffusion-geometry-bulk-flow

Release: https://github.com/TsutomuTakeuchiTTT/diffusion-geometry-bulk-flow/releases/tag/v0.1.0

The Zenodo record, not the presence of this metadata, establishes public availability.

Registration-stage note: the DOI is reserved and the GitHub repository has been created, but the version tag and Zenodo publication have not yet been confirmed. The date 2026-09-23 is the planned first-release date and must be checked before publishing.
<!-- RELEASE-IDENTIFIERS:END -->

This repository accompanies *Diffusion Geometry for Galaxy Bulk-Flow Reconstruction I: Foundations and Full-Vector Validation* and *II: Three-Dimensional Potential-Flow Reconstruction from Radial Velocities*. It contains the research sources, selected saved numerical outputs, configuration evidence, paper-to-code mapping, and small executable checks. It is **not** a validated real-survey analysis product or a new benchmark against other reconstruction methods.

## Start here

A self-contained demonstration is included. No cosmological catalog, historical cache, access token, or network connection is needed after Python dependencies are installed. It generates independent synthetic point geometries and runs the actual archived kernel, eigensystem, metric-calibrated gradient, Cartesian and radial-potential functions. Its in-span closure is an algebraic test, not the physical performance experiment reported in the papers.

From this directory, using Python 3.11 or later:

```sh
python -m pip install -r requirements.txt
python tools/verify_release.py
python tools/demo.py --output outputs/demo
python tools/recompute_capacity.py --output outputs/capacity
python tools/export_table_review.py --output outputs/paper_table_review.csv
```

These commands respectively verify the distribution, execute 90 small numerical component checks, recompute 27 capacity decisions from saved curves, and export the archived 373-cell manuscript/CSV comparison. Output directories/files must be new; rerunning with the same output name deliberately fails instead of overwriting results. Nothing is uploaded.

For a saved-data plot, run:

```sh
python tools/redraw_saved_mock1.py --output outputs/paperI_figure06
```

This executes the recovered Paper-I Mock-1 scale-comparison plotting script against bundled CSVs. It does not refit the field. Other figure families remain mapped to their research scripts; **this release does not promise one-command regeneration of all 31 final manuscript figures**.

## Environments and tests

The above numerical paths were tested in the existing Linux/Python 3.13.5 environment recorded in `reports/rc1_release_validation.json`. Direct dependency versions are pinned in `requirements.txt`. A fresh isolated installation was attempted, but package-index DNS was unavailable in the preparation environment. A successful clean installation, Windows execution, and the complete dependency-version matrix are **not** claimed.

For a separate environment, create a virtual environment and use its Python to install the requirements. Do not replace an existing scientific-work environment solely to run these checks. Unit tests additionally use PyYAML:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

See [the Japanese quick start](README_ja.md) for Notebook commands and [reproducibility](docs/reproducibility.md) for a precise separation of tested and archived paths.

## Contents

| Directory | Contents and role |
|---|---|
| `research_sources/` | 22 selected research scripts, including upstream radial-selection sources. Only private default paths were replaced with relative defaults; mathematical function bodies were retained. |
| `tools/` | Separate, guarded commands for the synthetic demo, distribution checks, saved-capacity recomputation, table-review export and one saved-data redraw. |
| `results/saved/` | 174 saved CSV/JSON/TXT files. These are paper-run outputs/records, not outputs of the small demo. 27 files have private path text redacted. |
| `configs/` | Explicit paper-specific conventions, including the 0.50 control-pool split and the noise-only plateau floor. |
| `metadata/` | Original/release hashes, numeric-cell provenance, configuration evidence and mappings for all 47 paper figures/tables. |
| `reports/` | Current release checks and clearly labeled historical audits, with their scopes. |

The three actual cosmological catalogs, large caches, simulation snapshots, manuscript PDFs, font files and private collection manifests are **not** distributed. They are not required for the default quick start. Their schema and checksums are documented for researchers who already have authorized copies. See [data scope](docs/data_scope.md).

## Scientific conventions retained

The primary graph uses 48 neighbors, bandwidth rank 16, multiplier 1, symmetric `maximum(K,K.T)`, diagonal 1, and alpha 0. Radial Mocks 2/3 exclude control IDs shared with either survey, then use a reference fraction of **0.50** and seed **20261201**, not the generic helper's 0.70 default. In Mock-1 noise experiments the plateau search starts at **514 parameters**; the raw and one-standard-error searches retain the full grid starting at 3. Small-demo generator normalization is distinct from the fixed 511-mode reference used by the paper's high-mode branches.

Archived missing coverage entries and zero SD/SEM conventions must not be interpreted as measured zero uncertainty. Two Paper-I Table-II last-digit rounding differences are preserved and explained in [the numerical notes](docs/numerical_notes.md). No result was altered to force agreement with a rounded manuscript cell.

## Running production research scripts

**Do not import all files in `research_sources/`.** Several execute a large analysis at module level and depend on prior notebook globals, legacy outputs or binary caches. They are provided as research-source records, not as independently installable modules. The executable quick start selects explicit function definitions and does not run those top-level drivers. See [production paths](docs/production_paths.md) before using the archived branches. Full 2048/4352-mode sweeps and full historical-cache equality were not rerun for this distribution.

## Citation and licensing

The confirmed software authors are **Tsutomu T. Takeuchi, Gayathri Asok, Hai-Xia Ma, and Yu Ogane**, in that order. Their ORCIDs and affiliations are recorded in `CITATION.cff`. Both papers should cite the same **version DOI** for this release, not two independently registered copies of the software.

Code and documentation are licensed under **BSD-3-Clause** (`LICENSE`). Saved numerical records and their source-provenance metadata are licensed under **CC BY 4.0** (`results/LICENSE`). These component-specific licenses do not apply to omitted original catalogs or third-party packages. See `THIRD_PARTY_NOTICES.md`.

The release metadata are prepared for a **single manually deposited Zenodo software record**. A DOI is reserved in the maintainer's account before it is inserted into the frozen archive. Do not also enable automatic GitHub-to-Zenodo archiving for this version. The existence of citation metadata is not proof that the record has been published. See [publication steps](docs/release_steps.md) and [the registration worksheet](docs/zenodo_registration_ja.md).

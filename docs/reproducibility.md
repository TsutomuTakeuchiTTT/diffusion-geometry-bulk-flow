# Reproducibility boundaries

This release has three different kinds of evidence. Do not combine them into a claim of complete paper reproduction.

1. **Archived paper results.** Saved CSVs and settings are distributed. Earlier audits compared 373 printed cells (371 matching at printed precision and two documented rounding differences), reaggregated 3,564 finite summary values, and reconstructed paper partitions/support from privately supplied catalogs. Historical audit JSONs state exactly which portions were checked. These numbers are not new production runs of this release.
2. **Executable result-level checks.** `recompute_capacity.py` recalculates all 27 saved aggregate capacity choices. It loads the original branch-specific plateau function and compares four capacity columns. `export_table_review.py` exports the existing numeric-cell comparison and checks that its named sources exist; it does not independently repeat the earlier row identification or bootstrap.
3. **Executable numerical components.** `demo.py` builds fresh synthetic point geometries and executes the original mathematical functions for kernel, Markov modes, calibrated gradients, Nyström extension, Cartesian fitting and radial potential fitting. No hidden vector truth enters its noisy radial validation. Exact in-span labels and closure are implementation tests, not cosmological field-accuracy estimates.

`redraw_saved_mock1.py` redraws one verified saved-data figure family using the original plotting driver. The output summary is compared numerically with saved values during packaging. It is a new rendering, not pixel-for-pixel archival recovery.

## What this release does not guarantee

There is no complete one-command reexecution of every high-mode production sweep. Some historical drivers require omitted catalogs, predecessor tables, notebook namespace state or large cached gradient bases. Their dependencies are described, rather than filled with guessed defaults. The exact source revision used for every historical run could not be proven from the old metadata. The released source fingerprints identify this distribution, not a newly discovered historical commit.

The distribution does not include or claim real-catalog validation, survey distance errors, correlated measurement errors, general PDE/Hodge software, or comparison against every reconstruction method. No new physical science result is introduced by the packaging tests.

## Installation evidence

The fixed direct versions in `requirements.txt` describe the existing tested environment, not a historical lockfile. The preparation attempted a new isolated venv and a binary-only dependency installation. It failed because the package index could not be resolved from this container. Thus **fresh installation is not checked**; neither are Windows nor a cross-version matrix. The actual commands succeeded in the separately recorded existing environment. A GitHub workflow is provided to run the quick checks on a fresh runner after repository upload, before the release tag is published, but it has not yet executed on GitHub.

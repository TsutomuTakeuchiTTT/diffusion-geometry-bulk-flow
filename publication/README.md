# Manuscript availability and citation

The templates/ directory contains deliberately unbound `.in` templates, not ready-to-submit paper text. No DOI has been invented. `tools/finalize_deposition.py` generates `software.bib` and `availability_PaperI.tex` / `availability_PaperII.tex` here only after the maintainer supplies the real reserved **software version DOI**, actual repository URL and publication date.

Use those paragraphs in the papers only after the Zenodo record is **published and publicly readable**. The templates disclose the omitted catalogs/snapshots/caches; they do not claim that all source data or all end-to-end workflows are publicly reproducible. No 'available on request' arrangement is invented.

Both papers use the same BibTeX key and software DOI. A paper/preprint DOI, a DOI for all software versions, and the specific software-version DOI are different identifiers. The `@misc` entry is suitable for BibTeX styles that lack an `@software` type.

This directory does not modify either manuscript. Existing local manuscript corrections, bibliography entries, figure labels and filenames remain untouched.

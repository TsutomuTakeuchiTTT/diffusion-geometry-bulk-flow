# Archived production branches

These scripts are source records and expert entry points, not a blanket guarantee that each can be independently run from this checkout. Their top-level code can start expensive computation and create/replace checkpoints. Use a disposable copy of a complete working directory, never the only copy of historical evidence.

Private default root literals have been replaced by relative `mock_data` paths. Where a script already supports an environment override, its original override name is preserved. Where it does not, the relative default is resolved from the working directory. No numerical formula or function definition was rewritten to simplify the release. All 22 selected files and 816 top-level function definitions were compared during packaging; see `metadata/source_manifest.json`.

| Paper branch | Archived source path | External prerequisites / scope |
|---|---|---|
| I, Mock 1 | `research_sources/paper_i/mock1_bulk_seed_scale_robustness_single_cell_v6.py`; `paperI_mock1_mode_number_convergence_to_2048.py` in the same directory | Catalog, legacy geometry and predecessor/local comparison outputs. The full continuation is not executed by the demo. |
| I, Mock 2 | `paperI_mock2_mode_convergence_to_4096.py`, `paperI_mock2_final_geometry_local_sensitivity.py` | Common-cell definitions in a shared namespace; selection-optimization prediction files, prior-grid records and cache inputs. |
| I, Mock 3 | `paperI_mock3_primary_mode_convergence_to_4096.py`, `paperI_mock3_highmode_loss_family_validation.py` | Common-cell definitions, octant prediction inputs, prior outputs and 4096-mode geometry. |
| II, Mock 1 | `research_sources/paper_ii/mock1_potential_flow_full_robustness_single_cell_v5.py` | Original catalogs/reference/cache; some comparison routes have a documented source fallback. |
| II, Mock 2 | `mock2_radial_potential_highmode_capacity_h3_resume_safe_v2.py`, `mock2_radial_potential_highmode_seed_robustness_v1.py` | Completed H3 gradient/design/cache metadata and original radial-selection predictions; resume-safe output names must not be assumed identical to the old H3 directory. |
| II, Mock 3 | `mock3_radial_potential_highmode_seed_robustness_v1_fixed_panel_d.py` | Completed H2 basis metadata, gradients/design and original octant predictions. The corrected categorical panel version is selected; the known faulty prior source is excluded. |
| II, noise | `mock1_potential_flow_measurement_noise_stage1b_seed_robustness_v1.py`, `mock1_potential_flow_heteroscedastic_noise_stage2_v1.py` | 2048-mode geometry, reference fields, full gradient cache and nested noise settings. Plateau floor is explicitly 514. |

The two upstream radial-selection sources are included under `research_sources/paper_ii/upstream/`. The generic shared definition module is under `research_sources/common/`. Simply running independent `%run` commands is not a substitute for a common-namespace execution route where `_REQUIRED_COMMON_SYMBOLS` are checked. No untested shell recipe is advertised as a complete reproduction.

## Figures

`metadata/figure_table_manifest_reviewed.json` and `docs/figure_table_map.md` map all 47 figure/table entries to candidate sources or saved metadata. Figure numbering changed during development; a filename containing `figure4` produces current Paper-I Figure 6. File stems, not historical script numbers, determine this mapping.

Paper-I Fig. 1's recovered schematic is the historical Japanese version, not the final English v3. For I Figs. 10/11/16, saved summaries identify final input/output stems, but exact final rendering-source equivalence remains unproved. These are explicitly retained as limitations, not solved by renaming files. The default release verification does not render or check every final figure.

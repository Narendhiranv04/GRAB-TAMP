# Benchmark runs — what is tracked here, and what is not

This directory holds 60 GB locally and about 700 MB in git. The split is
deliberate, and this file says where the line is so that a missing file
reads as a decision rather than an accident.

## Tracked: the measurement record

For every trial in the frozen 320 and in every ablation arm:

| file | why it is kept |
|---|---|
| `fm_diagnostics/fm_call_001.json` | the model's raw response. **Irreplaceable.** Every replay arm in this branch recompiles from these, so without them no ablation can be reproduced at all. |
| `structural_sanitization.json` | the v3 document after dangling-reference repair |
| `functional_specification.json` | the compiled specification — where free-form wording became fixed vocabulary |
| `functional_requirement_graph.json` | G_F |
| `run_manifest.json` | the frozen search contract: `region_order_used`, `region_ranking`, `inspected_regions`, phase timing |
| `graph_grounding_result.json` | φ: G_F → G_O, and why it was incomplete when it was |
| `symbolic_problem.json` | what was handed to A* |
| `result.json` | terminal status and the independent symbolic validation |
| `*_scored.json`, `full_metrics.json`, `*.md` | scoring passes and their write-ups |

## Not tracked: regenerable perception output

Rendered RGB views (16k PNGs per arm), point clouds (7k PLYs),
`policy_mode_comparison.json` (646 MB per arm on its own), per-object
`properties.json`, detection dumps, execution videos and GIFs. All of it is
a deterministic function of the tracked records plus the scene configs, and
all of it is regenerable by re-running the arm with `--spec-source raw-replay`,
which makes no model calls.

## The runs

| directory | trials | what it is |
|---|---|---|
| `full320_s01`, `_s02`, `_s03` | 320 | the 10 × 32 live design in three parallel streams; **309 FM responses survive** |
| `FINAL_10x32_RESULTS/` | — | scoring of the above. **Cite `rescored_current_scorer.json`.** `final_10x32.json` is pre-fix and kept only for provenance |
| `order_fm`, `order_random`, `order_worst` | 309 each | inspection-order ablation, paired replay over the frozen responses |
| `INSPECTION_ORDER_ABLATION/` | — | the paired scoring of those three arms, with Wilcoxon signed-rank |
| `zs_arm` | 309 | zero-shot fallback on capability, relation, region |
| `zs4_top1` | 309 | the above plus the role layer at top-1 |
| `zs4_thresh` | 22 | margin-gated four-layer arm, abandoned; kept as provenance |
| `workshop_open_costs.json` | — | measured robot opening cost per region from the home position |

The 11 trials missing from 320 are `INFRASTRUCTURE_UNAVAILABLE`: the endpoint
died before the model was reached. They are not observations of the method
and are excluded from every denominator, which is why the scorable count is
309 everywhere.

Directories prefixed `_discarded_` or `_aborted_` are kept on purpose. Each
one is a run that had to be thrown away — a dirty working tree during a live
benchmark, swap thrashing that contaminated the timing, a partial arm — and
the reason is worth more in the history than the disk space is.

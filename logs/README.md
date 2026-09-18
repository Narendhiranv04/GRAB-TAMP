# Distilled episode logs

`runs/` is gitignored (see `.gitignore`) and stays outside version control: the
OWL-TAMP replanning run alone is 160 MB, mostly rendered camera PNGs. This
directory holds the part of it that is worth reading and diffing -- the action
sequences and the per-episode planning record -- extracted from those artifacts.

| file | method | episodes | actions |
| --- | --- | --- | --- |
| `owl_tamp_action_sequences.jsonl` | OWL-TAMP | 374 | 1553 |
| `vlm_tamp_action_sequences.jsonl` | VLM-TAMP | 320 | 3417 |
| `robust_tamp_action_sequences.jsonl` | ROBUST-TAMP | 320 | 2859 |

Every file is JSON Lines: one object per episode. Each carries the benchmark
cell (`scene`, `variant`, `seed`, `expected_outcome`, `protocol`,
`camera_count`), the budget accounting (`model_calls`, `raw_vlm_requests`, and
`planning_rounds` or `replans`), the outcome (`status`, `success`,
`executed_actions`, `terminal_failure`), and `action_sequence[]` -- the ordered
actions, each with its success flag, failure code and symbolic effects.

An empty `action_sequence` is a real record, not a gap: those episodes failed at
planning and never reached execution.

## Conditions

Filter any file on `condition`. OWL-TAMP carries four, because it is the only
method run under more than one protocol:

| condition | scene | episodes | reported? |
| --- | --- | --- | --- |
| `owl_tamp_singleshot_final_20260913` | Kitchen, Living Room | 220 | yes |
| `workshop_replanning_20260914` | Workshop | 50 | yes |
| `owl_tamp_singleshot_workshop_superseded` | Workshop | 100 | **no** |
| `living_room_replanning_smoke` | Living Room | 4 | **no** |

`vlm_tamp_final_20260913` and `robust_tamp_final_20260913` are each 320
episodes across all three scenes, all reported.

### The two conditions that are not reported

`owl_tamp_singleshot_workshop_superseded` predates the `fixture` grounding fix
in `owl_tamp_baseline/domain.py`. Before that fix `INSERT` and `FASTEN` were
absent from the grounded set, so no Workshop plan could reach the goal and PGC
is 0.0 by construction. It is kept for provenance and must not be read as a
Workshop result; `workshop_replanning_20260914` postdates the fix and is the
reported Workshop condition.

`living_room_replanning_smoke` is 4 episodes, L1/L2/L7/L10 at seed 0 -- wiring
validation for the replanning protocol. L1 and L2 cover the success path, L7
and L10 the failure path where a second cycle is truncated part-way through
constraint generation. n=1 per cell supports no metric.

## Provenance

Single-shot conditions come from `runs/*/execution/final_20260913` at
`images_3` only. OWL-TAMP and VLM-TAMP are extracted from
`episode_result.json` (`result.action_history`); ROBUST-TAMP from
`discovery_replanning_result.json` (`history`); the replanning run from
`replanning_trace.json` (`action_history`) under
`runs/workshop/execution/owl_replanning_20260914`.

Every reported condition reproduces Table IV's published E2E success at the
right `N_F`, which is the check that the extraction is faithful rather than
merely plausible:

| condition | Kitchen | Living Room | Workshop |
| --- | --- | --- | --- |
| OWL-TAMP single-shot | 0.0 (60) | 60.0 (60) | -- |
| OWL-TAMP replanning | -- | -- | 15.0 (40) |
| VLM-TAMP | 0.0 (60) | 80.0 (60) | 26.2 (80) |
| ROBUST-TAMP | 0.0 (60) | 88.3 (60) | 0.0 (80) |

Only `workshop_replanning_20260914` and `living_room_replanning_smoke` have a
`cycles[]` field, because only that protocol has more than one planning cycle
per episode. It records calls spent, observation revision, sketch length,
constraint count, whether constraint generation completed, and the refinement
result.

## Reading notes

**`arguments` is not uniform across files.** Each method's native structure is
preserved rather than flattened, because the names carry information a
positional list would lose. OWL-TAMP's replanning condition emits a positional
list (`["object_0001", "region_0001"]`); every other condition emits a named map
(`{"object_id": "object_0001", "region_id": "region_0001"}`). The `operator`
field *is* normalised -- it is spelled `operator`, `skill` and `name` in the
three sources.

**Decoding is not seeded.** `baseline_common/inference.py` sets
`temperature 0.6` and `top_p 0.95` and sends no `seed`. The `seed` field here
indexes scene construction and the planner RNG, not the sampler, so the same
`(variant, seed)` can yield different plans across runs. Do not treat a single
episode as reproducible.

Living Room replanning is configured differently from Workshop's: only the
Appendix A.1.1 backtracking is enabled, with `prune_gap_actions` and
`drop_unobserved_goals` deliberately off. See `owl_tamp_baseline/run_living_room.py`
for why.

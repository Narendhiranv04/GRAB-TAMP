# Distilled episode logs

`runs/` is gitignored (see `.gitignore`) and stays outside version control: the
OWL-TAMP replanning run alone is 160 MB, mostly rendered camera PNGs. This
directory holds the part of it that is worth reading and diffing -- the action
sequences and the per-episode planning record -- extracted from those artifacts.

| file | method | episodes | actions |
| --- | --- | --- | --- |
| `owl_tamp_replanning_action_sequences.jsonl` | OWL-TAMP | 54 | 327 |
| `vlm_tamp_action_sequences.jsonl` | VLM-TAMP | 320 | 3417 |
| `robust_tamp_action_sequences.jsonl` | ROBUST-TAMP | 320 | 2859 |

Every file is JSON Lines: one object per episode. Each carries the benchmark
cell (`scene`, `variant`, `seed`, `expected_outcome`, `protocol`,
`camera_count`), the budget accounting (`model_calls`, `raw_vlm_requests`, and
`planning_rounds` or `replans`), the outcome (`status`, `success`,
`executed_actions`, `terminal_failure`), and `action_sequence[]` -- the ordered
actions, each with its success flag, failure code and symbolic effects.

## Provenance

`vlm_tamp_action_sequences.jsonl` and `robust_tamp_action_sequences.jsonl` come
from `runs/*/execution/final_20260913`, the reported single-shot grid, at
`images_3` only: 120 Kitchen + 100 Living Room + 100 Workshop episodes each.
VLM-TAMP is extracted from `episode_result.json` (`result.action_history`),
ROBUST-TAMP from `discovery_replanning_result.json` (`history`).

Both reproduce Table IV's published E2E success exactly, which is the check that
the extraction is faithful:

| method | Kitchen | Living Room | Workshop |
| --- | --- | --- | --- |
| VLM-TAMP | 0.0 | 80.0 | 26.2 |
| ROBUST-TAMP | 0.0 | 88.3 | 0.0 |

`owl_tamp_replanning_action_sequences.jsonl` holds two conditions.
`workshop_replanning_20260914` is 50 episodes, W1-W10 x seeds 0-4, protocol
`replanning` at a 15-call budget -- the reported Workshop OWL-TAMP condition
(40 feasible, 10 infeasible). `living_room_replanning_smoke` is 4 episodes,
L1/L2/L7/L10 at seed 0: **wiring validation only, not a reported result.** L1
and L2 cover the success path, L7 and L10 the failure path where a second cycle
is truncated part-way through constraint generation. n=1 per cell supports no
metric.

Only the OWL-TAMP file has a `cycles[]` field, because only that protocol has
more than one planning cycle per episode. It records calls spent, observation
revision, sketch length, constraint count, whether constraint generation
completed, and the refinement result.

## Reading notes

**`arguments` is not uniform across files.** Each method's native structure is
preserved rather than flattened, because the names carry information a
positional list would lose. OWL-TAMP emits a positional list
(`["object_0001", "region_0001"]`); VLM-TAMP and ROBUST-TAMP emit a named map
(`{"object_id": "object_0001", "region_id": "region_0001"}`). The `operator`
field *is* normalised -- it is spelled `operator` in OWL-TAMP's source, `skill`
in VLM-TAMP's and `name` in ROBUST-TAMP's.

**Decoding is not seeded.** `baseline_common/inference.py` sets
`temperature 0.6` and `top_p 0.95` and sends no `seed`. The `seed` field here
indexes scene construction and the planner RNG, not the sampler, so the same
`(variant, seed)` can yield different plans across runs. Do not treat a single
episode as reproducible.

**Workshop OWL-TAMP single-shot data is not here and should not be used.**
`final_20260913` predates the `fixture` grounding fix in
`owl_tamp_baseline/domain.py`, so `INSERT` and `FASTEN` were absent from the
grounded set and no Workshop plan could reach the goal (PGC 0.0). The
replanning run postdates the fix.

Living Room replanning is configured differently from Workshop's: only the
Appendix A.1.1 backtracking is enabled, with `prune_gap_actions` and
`drop_unobserved_goals` deliberately off. See `owl_tamp_baseline/run_living_room.py`
for why.

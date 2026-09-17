# Distilled episode logs

`runs/` is gitignored (see `.gitignore`) and stays outside version control: the
OWL-TAMP replanning run alone is 160 MB, mostly rendered camera PNGs. This
directory holds the part of it that is worth reading and diffing -- the action
sequences and the per-cycle planning record -- extracted from those artifacts.

## `owl_tamp_replanning_action_sequences.jsonl`

One JSON object per episode, 54 episodes.

| field | meaning |
| --- | --- |
| `condition` | which run the episode came from (see below) |
| `variant`, `seed`, `expected_outcome` | benchmark cell |
| `max_model_calls`, `model_calls` | whole-episode budget and what was spent |
| `planning_cycles`, `replans`, `skeletons_tested` | protocol accounting |
| `status`, `success`, `executed_actions`, `failure` | episode outcome |
| `cycles[]` | per planning cycle: calls spent, observation revision, sketch length, constraint count, whether constraint generation completed, and the refinement result |
| `action_sequence[]` | the ordered actions, each with its originating planning cycle, success flag, failure code and symbolic effects |

### Conditions

`workshop_replanning_20260914` -- 50 episodes, W1-W10 x seeds 0-4, protocol
`replanning`, 15-call budget, 3 cameras, `model-native` decoding. This is the
reported Workshop OWL-TAMP condition (40 feasible, 10 infeasible). Source:
`runs/workshop/execution/owl_replanning_20260914`.

`living_room_replanning_smoke` -- 4 episodes, L1/L2/L7/L10 at seed 0, same
protocol and budget. **Wiring validation only, not a reported result.** L1 and
L2 exercise the success path (`GOAL_COMPLETE`, 0 replans); L7 and L10 exercise
the failure path (`MODEL_BUDGET_EXHAUSTED` after a second cycle is truncated
mid-constraint-generation). n=1 per cell, so these support no metric.

## Reading notes

Decoding is not seeded: `baseline_common/inference.py` sets `temperature 0.6`
and `top_p 0.95` and sends no `seed`. The `seed` field here indexes scene
construction and the planner RNG, not the sampler, so the same `(variant, seed)`
can yield different sketches across runs. Do not treat a single episode as
reproducible.

Living Room replanning is configured differently from Workshop's: only the
Appendix A.1.1 backtracking is enabled, with `prune_gap_actions` and
`drop_unobserved_goals` deliberately off. See `owl_tamp_baseline/run_living_room.py`
for why.

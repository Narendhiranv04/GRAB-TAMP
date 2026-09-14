# FM-guided Functional TAMP — how the pipeline works, and how to run it

Reference for the working system as it stands at `caec01bc`. Written so a later
change can be aimed at the right stage instead of a plausible-looking one.

---

## 1. What a single trial does

One trial is one (domain, variant) pair taken from the FM's description of the
task to a validated symbolic plan. Exactly **one FM call** is made, at the start;
everything after it is deterministic given that response.

```
instruction + 3 rendered RGB views
        │
        ▼  ONE FM call  (qwen35-9b, schema v3, temp 0.6)
raw response          fm_diagnostics/fm_call_001.json
        │
        ▼  structural sanitizer            structural_sanitization.json
        │  v3 → canonical document, dangling references repaired
        ▼  semantic compilation            functional_specification.json
        │  ── this is where free-form FM wording becomes fixed vocabulary ──
        │     roles        coffee_cup      → coffee_container
        │     capabilities "pour"          → TRANSFER_CONTENT_TO_CONTAINER
        │     relations    "placed_on"     → FITS_SET_ON
        │     regions      "table drawer"  → D1
        ▼  G_F  functional requirement graph
        │
        ▼  search contract frozen          run_manifest.json: region_order_used
        │  region inspection order fixed before any perception
        ▼  perception + region search      observed_search/, observed_scene_graph.json
        │  YOLO-World open-vocab semantics → point-cloud geometry → G_O
        ▼  grounding  φ: G_F → G_O         graph_grounding_result.json
        │
        ▼  ≤ 1 A* invocation               symbolic_problem.json
        ▼  independent symbolic validation result.json
```

The four canonicalization vocabularies are fixed and small:

| layer | target vocabulary | size |
|---|---|---|
| role | `coffee_container`, `soup_eating_utensil`, `driver`, `fastener`, … | 6 / 6 / 3 per domain |
| capability | `STIR_COFFEE`, `TRANSFER_CONTENT_TO_CONTAINER`, `FASTEN_JOINT`, … | 7 total |
| relation | `INSERTABLE_IN`, `REACHES_BOTTOM`, `FITS_SET_ON`, … | 9 binary |
| region | `D1 D2 C1 C2 B1`, `LEFT_DRAWER RIGHT_DRAWER TOOL_CABINET` | 5 / 0 / 3 |

Living Room declares **no inspectable regions** — everything is visible from the
start, so it has no search stage and no recovery from a bad initial grounding.

---

## 2. Where a trial can fail, in pipeline order

First-cause attribution assigns each failing trial to the **earliest** stage that
broke; that ordering is why the later stages look small.

| stage | recorded as | what it means |
|---|---|---|
| Task Specification | `VLM_SPEC_FAILED` | no usable FM document |
| Graph Compilation | `unresolved_roles`, `unresolved_semantics`, `disabled_groups`, `unresolved_required_operations` | FM output could not become an executable graph |
| Object Discovery | search exhausted, grounding incomplete | region opened, object never found |
| Functional Assignment | grounding incomplete, search not exhausted | objects found, no consistent role assignment |
| Planning | grounding complete, goals unmet | A* found no complete sequence |

Measured on the frozen 320 (192 feasible, 83 succeeding): **graph compilation is
the dominant blocker**, and its two largest sub-causes are role collisions
(`AMBIGUOUS_ROLE_MAPPING`, 75) and unmapped operations.

---

## 3. Running it

All commands from the repository root with `PYTHONPATH=.`.

### One trial

```bash
PYTHONPATH=. python3 -m mujoco_scenes.functional_tamp_pipeline.run \
  --domain kitchen --variant K1 --mode vlm --dry-run \
  --output-root benchmark_reports/scratch
```

`--mode gt` uses the ground-truth specification provider instead of the FM.
`--search-order {auto,oracle,provider,random,fixed}` selects the region order;
`random` additionally requires `--search-seed N`. `--dry-run` opens containers by
setting the simulator joint directly; without it, workshop uses robot-actuated
opening.

### The full 32-variant matrix

```bash
PYTHONPATH=. python3 scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm --spec-source live \
  --output-root benchmark_reports/<run-name>
```

`--spec-source` decides where the specification comes from, and this is the
distinction that matters most for reproducibility:

| value | source | FM called? |
|---|---|---|
| `live` | a fresh FM call per trial | **yes** |
| `replay` | the archived `functional_specification.json` | no |
| `raw-replay` | the archived `fm_diagnostics/fm_call_001.json` | no |

**`raw-replay` is the one to use for any change to canonicalization**, because it
re-runs compilation from the model's own words. `replay` loads the already
compiled specification and would carry the old mapping forward unchanged.

### Repeated runs (the 10 × 32 design)

```bash
PYTHONPATH=. python3 scripts/run_live_repeat_experiment.py \
  --output-root benchmark_reports/<run-name> --repeats 10 \
  --base-url http://127.0.0.1:8000/v1 --model qwen35-9b
```

This owns the frozen sampler — nine keys held fixed across every trial. Import
`SAMPLER` from it rather than restating them; an earlier runner that redeclared
only `max_tokens` and `enable_thinking` dropped `presence_penalty`, the model ran
away, and two arms became incomparable.

### Scoring

```bash
PYTHONPATH=. python3 scripts/score_full_experiment.py \
  --root benchmark_reports/<run-name> --out benchmark_reports/<run-name>
```

Writes `full_metrics.json` with per-trial rows and a summary. Goal coverage and
functional assignment coverage are scored against the frozen ground truth in
`GT_GOAL_COVERAGE/` and `GT_VALID_ROLE_ASSIGNMENTS/`.

### Serving the model

Only `--spec-source live` needs it. Everything else reads archived responses.

```bash
ssh -i ~/keyfile -p 27617 -N -L 8000:127.0.0.1:8000 long-horizon@0.tcp.in.ngrok.io
curl -s http://127.0.0.1:8000/v1/models        # expect qwen35-9b
```

---

## 4. Frozen artifacts

| path | what it holds |
|---|---|
| `benchmark_reports/full320_s01,_s02,_s03/` | the 320-trial run, three parallel streams; **309 archived FM responses** survive |
| `benchmark_reports/FINAL_10x32_RESULTS/rescored_current_scorer.json` | the 320 rescored with the current scorer — **cite this** |
| `benchmark_reports/FINAL_10x32_RESULTS/final_10x32.json` | the original scoring pass, kept for provenance; carries pre-fix FAC and goal coverage |
| `GT_GOAL_COVERAGE/`, `GT_VALID_ROLE_ASSIGNMENTS/` | ground truth for the two scored metrics |
| `benchmark_reports/order_fm,_random,_worst/` | inspection-order ablation, 309 trials each |
| `benchmark_reports/zs4_top1/` | zero-shot canonicalization arm, 309 trials |

11 of the 320 are `INFRASTRUCTURE_UNAVAILABLE` — the endpoint died before the
model was reached. They are not observations of the method and are excluded from
every denominator, which is why the scorable count is 309.

---

## 5. Changing it without changing it

Every ablation in this repository runs as a **shadow**: a context manager patches
named functions in process, runs the unmodified evaluator, and restores them on
exit. No pipeline file is edited, and the arm can be run against the frozen
responses with no FM calls.

| shadow | patches | arm |
|---|---|---|
| `mujoco_scenes/fm_search_order_shadow.py` | `run.search_until_satisfied`, the domain `run_to_plan` functions | inspection order + phase timing |
| `mujoco_scenes/fm_evidence_ablation.py`, `fm_ablation_shadow.py` | `grounding.ground_graph` | withhold semantic / unary / binary evidence |
| `mujoco_scenes/fm_zs_canonicalization_shadow.py` | the four canonicalization resolvers | zero-shot fallback |

The pattern that keeps a shadow honest:

- **Augment, never replace.** Consult the existing resolver first and act only
  where it returns nothing, so a trial the current system handles takes the
  identical path and the arm cannot regress what already works.
- **Report the counts.** A patch that silently fails to apply produces a second
  copy of the baseline and reads as a clean null result. Every shadow counts how
  often it fired and warns when it never did.
- **Verify the patch point on one trial before running 309.** Twice a plausible
  function turned out not to be the one that decides: `_map_role` is overridden
  by `resolve_role_type_hypotheses`, and `_participant_slot_fit` is never reached
  because `interpret_operation` rejects the operation earlier on endpoint
  signature. Both cost hours and both were visible in a single trial.

---

## 6. Known open problems

**Role collisions.** `AMBIGUOUS_ROLE_MAPPING` (75 occurrences) is two FM roles
resolving to the same canonical role. An augment-only fallback never fires for
it, because the mapper already returned an answer.

**One FM role, two canonical roles.** Living Room's most common failure: the model
writes a single `placement_surface` whose stated function supports both the
refreshments and the remote. It canonicalizes to `PERSONAL_CUP_SAUCER_REGION`,
and `SUPPORT_ENTERTAINMENT_CONTROL` then finds no capability whose target
signature accepts it, so the remote placement is dropped — 10 of 60 feasible
Living Room trials. Fixing it means role resolution becoming **operation
relative** rather than one canonical name per FM role, which is a design change
rather than a patch.

**`NO_GLOBAL_REGION_ASSIGNMENT`.** 8 feasible Living Room trials are declared
infeasible with this message, which says what failed but not which constraint
could not be met.

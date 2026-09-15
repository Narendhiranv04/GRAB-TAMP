# GRAB-TAMP

Grounded functional task and motion planning from a single foundation-model call.

A robot is given a task in natural language and three rendered views of a scene,
and must work out what the task requires, which objects can fill those roles,
where to look for the ones it cannot see, and what sequence of actions achieves
the goal — without being told which object is which. GRAB-TAMP makes **one**
foundation-model call. The response is sanitized and compiled into a functional
requirement graph over fixed vocabularies of roles, capabilities, relations and
regions, with a zero-shot NLI fallback for wording the cue tables do not cover.
The region inspection order is then frozen before any perception runs;
open-vocabulary detection and point-cloud geometry build an observed scene
graph; a joint grounding binds roles under semantic *and* geometric
constraints; and one A\* call produces an action sequence that an independent
validator checks. The model is never consulted again, so no failure can be
repaired by resampling it.

## Setup

Python 3.13.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-zs.txt
python3 mujoco_scenes/scripts/prepare_semantic_models.py    # detector + CLIP weights
```

Install both requirement files in one `pip` call, as above: separate calls can
resolve torchvision against the wrong torch. On a headless machine set
`MUJOCO_GL=egl`. If `python3 -m venv` reports that `ensurepip` is unavailable,
either install the distribution's `python3-venv` package or use conda
(`conda create -n grab python=3.13`).

| component | model |
|---|---|
| foundation model | `qwen35-9b`, OpenAI-compatible endpoint (only for `--live`) |
| open-vocabulary detector | YOLO-World `yolov8m-worldv2.pt` |
| detector embedding | CLIP `ViT-B-32` |
| canonicalization fallback | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` (CPU) |

Replaying the archived responses needs **no GPU**. Serving the foundation model
for a `--live` run needs a GPU with room for a 9B VLM at 32k context.

## Run

```bash
./run_demo.sh                    # kitchen K3
./run_demo.sh workshop W2
./run_demo.sh living_room L1
```

Output for `./run_demo.sh`:

```
status          ACTION_SEQUENCE_READY
regions opened  ['C1', 'C2', 'D1', 'D2', 'B1']
role binding    {'coffee_container': ['object_0001', 'object_0002'],
                 'coffee_stirrer': 'object_0004', 'water_source': 'object_0007', ...}
action sequence:
   1. PICK(object_0005)
   2. PLACE_SERVING_UTENSIL(object_0005, object_0003)
   ...
  19. PICK(object_0002)
  20. PLACE(object_0002, dining_table)
```

```bash
./run_benchmark.sh               # regenerate Tables III and IV (seconds)
./run_benchmark.sh --replay      # re-run all 32 variants from archived responses
./run_benchmark.sh --live        # fresh model calls
```

## Domains and variants

32 variants across three MuJoCo domains, 10 repeats each — **320 trials**.

| domain | variants | feasible | infeasible | inspectable regions |
|---|---|---|---|---|
| kitchen | K1–K12 | K1–K6 | K7–K12 | 5 |
| living_room | L1–L10 | L1–L6 | L7–L10 | 0 |
| workshop | W1–W10 | W1–W8 | W9, W10 | 3 |

Infeasible variants admit no valid grounding; the correct behaviour is to
reject them, which is what the correct-rejection metric measures. Living Room
has no inspectable regions — everything is visible from the start — so it has
no search stage.

## Results

`./run_benchmark.sh` regenerates both tables into `results/tables/paper_tables.md`.

**Table III — end to end.** PGC is mean plan goal coverage; E2E is the share of
feasible trials satisfying every reference goal; CR is correct rejection.

| Domain | N_F | PGC | E2E succ. | N_I | CR |
|---|---:|---:|---:|---:|---:|
| Kitchen | 60 | 82.8 | 65.0 | 60 | 50.0 |
| Living Room | 60 | 48.7 | 31.7 | 40 | 70.0 |
| Workshop | 80 | 69.6 | 62.5 | 20 | 20.0 |
| **Overall** | **200** | **67.3** | **54.0** | **120** | **51.7** |

**Table IV — verification components.** Reference role and relation slots
classified correct / wrong / missing (share of functional assignment coverage).

| Scene | Verification | Correct | Wrong | Missing |
|---|---|---:|---:|---:|
| Kitchen | Semantic only | 96 (12.3) | 19 (2.4) | 665 (85.3) |
| | + Unary | 103 (13.2) | 21 (2.7) | 656 (84.1) |
| | **+ Binary** | **514 (65.9)** | 63 (8.1) | **203 (26.0)** |
| Living Room | Semantic only | 141 (18.1) | 1 (0.1) | 638 (81.8) |
| | + Unary | 96 (12.3) | 3 (0.4) | 681 (87.3) |
| | **+ Binary** | **439 (56.3)** | 20 (2.6) | **321 (41.2)** |
| Workshop | Semantic only | 86 (17.9) | 13 (2.7) | 381 (79.4) |
| | + Unary | 47 (9.8) | 8 (1.7) | 425 (88.5) |
| | **+ Binary** | **239 (49.8)** | 55 (11.5) | **186 (38.8)** |
| **Overall** | Semantic only | 323 (15.8) | 33 (1.6) | 1684 (82.5) |
| | + Unary | 246 (12.1) | 32 (1.6) | 1762 (86.4) |
| | **+ Binary** | **1192 (58.4)** | 138 (6.8) | **710 (34.8)** |

**Table IV — inspection order.** Fixed order is a privileged worst case that
reads which regions are empty from the scene configuration and opens those
first. Total time is region-opening cost plus grounding time.

| Scene | Order | Regions inspected | Candidate checks | Total time (s) |
|---|---|---:|---:|---:|
| Kitchen | Fixed | 4.33 | 134.6 | 82.94 |
| | **FM-ranked** | **3.91** | **134.6** | **76.22** |
| Living Room | Fixed | 0.00 | 23.2 | 6.77 |
| | **FM-ranked** | **0.00** | **23.2** | **6.77** |
| Workshop | Fixed | 2.58 | 14.2 | 287.64 |
| | **FM-ranked** | **2.32** | **13.4** | **268.20** |

## Ground-truth executions

Recorded robot executions of the reference plans, used as the action-sequence
reference for scoring. Full sequences in `ground_truth/execution_reference.json`.

| variant | actions | exec. (s) | | variant | actions | exec. (s) |
|---|---:|---:|---|---|---:|---:|
| kitchen/K1 | 24 | 526.7 | | living_room/L4 | 19 | 263.5 |
| kitchen/K2 | 27 | 618.9 | | living_room/L5 | 16 | 234.0 |
| kitchen/K3 | 25 | 657.1 | | living_room/L6 | 20 | 279.7 |
| kitchen/K4 | 28 | 708.1 | | workshop/W1 | 6 | 390.3 |
| kitchen/K5 | 26 | 542.7 | | workshop/W2 | 6 | 380.3 |
| living_room/L1 | 20 | 278.0 | | workshop/W3 | 7 | 545.6 |
| living_room/L2 | 16 | 225.3 | | workshop/W4 | 7 | 499.3 |
| living_room/L3 | 20 | 272.1 | | | | |

Example — `workshop/W1`:

```
OPEN(LEFT_DRAWER)
PICK(workshop_medium_phillips_screw, LEFT_DRAWER)
PLACE(workshop_medium_phillips_screw, workshop_frame_joint)
PICK(workshop_long_phillips_driver, LEFT_DRAWER)
SCREW(workshop_long_phillips_driver, workshop_medium_phillips_screw, workshop_frame_joint)
PLACE(workshop_long_phillips_driver, MAIN_WORKBENCH_ZONE)
```

## Layout

```
mujoco_scenes/functional_tamp_pipeline/   the method
mujoco_scenes/{assets,configs}/           the three scenes and the 32 variants
mujoco_scenes/fm_*_shadow.py              in-process shadows: the zero-shot
                                          canonicalization fallback, and the
                                          evidence and inspection-order ablations
scripts/                                  evaluation, scoring, table generation
ground_truth/                             evaluation-only reference data
data/reference_run/                       32 archived responses and their outcomes
data/metrics/                             the aggregates the tables are built from
experiments/alias_resolution_study/       zero-shot canonicalization study
```

Ground truth is read only by the offline scoring scripts, never at runtime:

```bash
PYTHONPATH=. python3 scripts/audit_no_gt_leakage.py
```

The audit fails if the pipeline reads ground-truth feasibility, identities,
roles, success, inventories or expected answers, or branches on a variant name
or trial index.

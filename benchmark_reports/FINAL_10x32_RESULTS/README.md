# Final 10×32 Experiment — Corrected Metrics and Failure Analysis

Live run, fresh FM call per trial, **no prompt variance**.
Model `qwen35-9b` rev `baf3755e2be6`, prompt hash `156c661a45a4`.

320 trials written · **309 scorable** · 11 excluded as infrastructure outages.
An excluded trial never reached the model, so it is not an observation of the
method and is removed from every denominator.

> **Rescored 2026-09-14 against the current scorer.** The numbers below come
> from re-running `score_full_experiment.py` over all 320 archived trials
> (`rescored_current_scorer.json`, 320/320 rescored, no errors). An earlier
> revision of this file reported goal coverage from a scoring pass that predated
> the Living Room cup/saucer correction, which had counted correct
> shared/distinct placements as failures. Goal coverage was **0.455**; it is
> **0.580**, and Living Room moves from 0.087 to 0.487. FAC, end-to-end,
> regions inspected and false completion are unchanged.

## Headline

| metric | scope | value |
|---|---|---|
| **1. Goal coverage** | feasible | **0.580** (775/1182 goals) · 0.795 (replay-valid, 140/192) |
| **2. Functional assignment coverage** | feasible | **0.589** · slots 1192/1492 = 79.9% |
| **3. False completion** | infeasible | **52/117 = 44%** · claimed-complete **1** |
| **4. End-to-end** | feasible | **80/192 = 42%** any order · 47/192 = 24% exact order |
| **5. Regions inspected** | feasible 1.90 · infeasible 2.67 | kitchen 3.99 · living_room 0.00 · workshop 2.32 |
| **6. Objects semantically + geometrically checked** | both | kitchen 7.9 · living_room 9.1 · workshop 5.5 |

Infeasible trials inspect **more** regions than feasible ones (2.67 against
1.90): when the required object does not exist, the search exhausts the scene
before giving up.

### Success rates

| definition | rate |
|---|---|
| feasible — all goals met | 83/192 = 43% |
| feasible — end-to-end action match | 80/192 = 42% |
| feasible — **both** | **80/192 = 42%** |
| infeasible — correctly rejected | 65/117 = 56% |
| **whole benchmark (strict)** | **145/309 = 47%** |

Matching the GT action multiset is currently *sufficient but not necessary*,
which is the reverse of what the pre-correction scoring showed: all 80 runs
that match the actions also satisfy every goal, while 3 runs satisfy every goal
by a different action sequence. The earlier reading was an artefact of the
Living Room cup/saucer defect suppressing goal credit.

---
## Domain-wise

| domain | trials | feasible | goal cov. | FAC | end-to-end | rejected | false compl. |
|---|---:|---:|---:|---:|---:|---:|---:|
| **kitchen** | 113 | 54 | 0.787 | 0.732 | 33/54 = 61% | 34/59 = 58% | 25/59 |
| **living_room** | 99 | 60 | 0.487 | 0.563 | 19/60 = 32% | 27/39 = 69% | 12/39 |
| **workshop** | 97 | 78 | 0.509 | 0.511 | 28/78 = 36% | 4/19 = 21% | 15/19 |

---
## Variant-wise

### kitchen

| variant | n | type | goal cov. | FAC | end-to-end | rejected | false compl. | dominant status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| K1 | 10 | feasible | 0.667 | 0.662 | 6/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| K2 | 8 | feasible | 0.979 | 0.981 | 7/8 | — | — | `ACTION_SEQUENCE_READY` |
| K3 | 9 | feasible | 0.824 | 0.795 | 4/9 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| K4 | 10 | feasible | 0.525 | 0.554 | 2/10 | — | — | `ACTION_SEQUENCE_READY` |
| K5 | 7 | feasible | 0.976 | 0.637 | 6/7 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| K6 | 10 | feasible | 0.850 | 0.792 | 8/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| K7 | 10 | infeasible | — | — | — | 7/10 | 3/10 | `NO_MEANINGFUL_CANDIDATE_PLAN` |
| K8 | 10 | infeasible | — | — | — | 3/10 | 7/10 | `EXHAUSTED_NO_VALID_GROUNDING` |
| K9 | 10 | infeasible | — | — | — | 8/10 | 2/10 | `NO_MEANINGFUL_CANDIDATE_PLAN` |
| K10 | 10 | infeasible | — | — | — | 5/10 | 5/10 | `NO_MEANINGFUL_CANDIDATE_PLAN` |
| K11 | 9 | infeasible | — | — | — | 4/9 | 5/9 | `EXHAUSTED_NO_VALID_GROUNDING` |
| K12 | 10 | infeasible | — | — | — | 7/10 | 3/10 | `NO_MEANINGFUL_CANDIDATE_PLAN` |

### living_room

| variant | n | type | goal cov. | FAC | end-to-end | rejected | false compl. | dominant status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| L1 | 10 | feasible | 0.400 | 0.485 | 4/10 | — | — | `ACTION_SEQUENCE_READY` |
| L2 | 10 | feasible | 0.500 | 0.646 | 4/10 | — | — | `ACTION_SEQUENCE_READY` |
| L3 | 10 | feasible | 0.520 | 0.608 | 2/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| L4 | 10 | feasible | 0.500 | 0.500 | 5/10 | — | — | `ACTION_SEQUENCE_READY` |
| L5 | 10 | feasible | 0.540 | 0.638 | 2/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| L6 | 10 | feasible | 0.460 | 0.500 | 2/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| L7 | 10 | infeasible | — | — | — | 6/10 | 4/10 | `INFEASIBLE` |
| L8 | 10 | infeasible | — | — | — | 6/10 | 4/10 | `VLM_SPEC_FAILED` |
| L9 | 10 | infeasible | — | — | — | 6/10 | 4/10 | `VLM_SPEC_FAILED` |
| L10 | 9 | infeasible | — | — | — | 9/9 | 0/9 | `INFEASIBLE` |

### workshop

| variant | n | type | goal cov. | FAC | end-to-end | rejected | false compl. | dominant status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| W1 | 10 | feasible | 0.533 | 0.550 | 5/10 | — | — | `ACTION_SEQUENCE_READY` |
| W2 | 10 | feasible | 0.667 | 0.700 | 6/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| W3 | 10 | feasible | 0.233 | 0.533 | 0/10 | — | — | `EXHAUSTED_NO_VALID_GROUNDING` |
| W4 | 8 | feasible | 0.125 | 0.125 | 0/8 | — | — | `EXHAUSTED_NO_VALID_GROUNDING` |
| W5 | 10 | feasible | 0.200 | 0.217 | 0/10 | — | — | `EXHAUSTED_NO_VALID_GROUNDING` |
| W6 | 10 | feasible | 0.800 | 0.867 | 4/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| W7 | 10 | feasible | 0.567 | 0.400 | 5/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| W8 | 10 | feasible | 0.867 | 0.617 | 8/10 | — | — | `PARTIAL_ACTION_SEQUENCE_READY` |
| W9 | 10 | infeasible | — | — | — | 3/10 | 7/10 | `EXHAUSTED_NO_VALID_GROUNDING` |
| W10 | 9 | infeasible | — | — | — | 1/9 | 8/9 | `EXHAUSTED_NO_VALID_GROUNDING` |

---
## Failure statuses

**kitchen — feasible**

| status | n |
|---|---:|
| `PARTIAL_ACTION_SEQUENCE_READY` | 26 |
| `ACTION_SEQUENCE_READY` | 16 |
| `NO_MEANINGFUL_CANDIDATE_PLAN` | 8 |
| `EXHAUSTED_NO_VALID_GROUNDING` | 3 |
| `VLM_SPEC_FAILED` | 1 |

**kitchen — infeasible**

| status | n |
|---|---:|
| `NO_MEANINGFUL_CANDIDATE_PLAN` | 27 |
| `EXHAUSTED_NO_VALID_GROUNDING` | 23 |
| `VLM_SPEC_FAILED` | 7 |
| `PARTIAL_ACTION_SEQUENCE_READY` | 2 |

**living_room — feasible**

| status | n |
|---|---:|
| `PARTIAL_ACTION_SEQUENCE_READY` | 18 |
| `ACTION_SEQUENCE_READY` | 17 |
| `INFEASIBLE` | 9 |
| `VLM_SPEC_FAILED` | 8 |
| `NO_MEANINGFUL_CANDIDATE_PLAN` | 5 |
| `EXHAUSTED_NO_VALID_GROUNDING` | 3 |

**living_room — infeasible**

| status | n |
|---|---:|
| `INFEASIBLE` | 16 |
| `EXHAUSTED_NO_VALID_GROUNDING` | 10 |
| `VLM_SPEC_FAILED` | 7 |
| `NO_MEANINGFUL_CANDIDATE_PLAN` | 4 |
| `ACTION_SEQUENCE_READY` | 1 |
| `PARTIAL_ACTION_SEQUENCE_READY` | 1 |

**workshop — feasible**

| status | n |
|---|---:|
| `PARTIAL_ACTION_SEQUENCE_READY` | 24 |
| `ACTION_SEQUENCE_READY` | 18 |
| `EXHAUSTED_NO_VALID_GROUNDING` | 15 |
| `INFEASIBLE` | 10 |
| `VLM_SPEC_FAILED` | 10 |
| `NO_MEANINGFUL_CANDIDATE_PLAN` | 1 |

**workshop — infeasible**

| status | n |
|---|---:|
| `EXHAUSTED_NO_VALID_GROUNDING` | 13 |
| `INFEASIBLE` | 3 |
| `PARTIAL_ACTION_SEQUENCE_READY` | 2 |
| `VLM_SPEC_FAILED` | 1 |

---
## Scorer corrections applied

All were defects in offline scoring, not in the pipeline. No pipeline code was
changed and no trial was re-run.

| # | defect | effect |
|---|---|---|
| 1 | Identity resolver covered kitchen only, so workshop grounded ids never matched GT body names | workshop FAC 0% → 68.7% |
| 2 | Living Room GT asked for `cup` and `saucer` as separate roles; the detector emits `cup_or_saucer` and the grounder binds a `CUP_SAUCER_SET`, so those slots were unsatisfiable by construction | LR GT rewritten to the grounded roles |
| 3 | Relational extractor emitted `personal_set_on_region` where the GT declares `set_on_personal_region`, and ignored seat context and remote placement | LR relational slots 0 → 169 |
| 4 | Unlabelled clusters (8.4% of kitchen observations) had no category to rank within | kitchen unresolved 9.3% → 1.7% |

Before these fixes overall FAC read 0.435; it is **0.589**. Correction 2 also
raised Living Room goal coverage from 0.087 to 0.487, which the previous
revision of this file did not carry.

## Caveats

- 11 trials lost to network outages; the denominator is 309, not 320.
- Metric 6 counts checks whose results were recorded in the observed scene
  graph, not internal call counts, which the pipeline does not instrument.
- Living Room inspects zero regions by design; never average metric 5 across domains.
- `final_10x32.json` is the original scoring pass and is retained for provenance.
  It carries the pre-fix workshop FAC (0.000) and goal coverage. Use
  `rescored_current_scorer.json` for any reported number.

# Results index — FM-guided functional TAMP

Every experiment on this branch, what it concluded, and where its data is.
Numbers here are read from the committed artifacts, not restated from memory;
each row names the file to check. For how the system works and how to run it,
see [PIPELINE.md](PIPELINE.md).

---

## 1. The main run — 320 trials, 10 × 32 live

`benchmark_reports/full320_s01,_s02,_s03/` ·
scored in `benchmark_reports/FINAL_10x32_RESULTS/rescored_current_scorer.json` ·
written up in [that directory's README](benchmark_reports/FINAL_10x32_RESULTS/README.md)

Fresh FM call per trial, `qwen35-9b` rev `baf3755e2be6`, prompt hash
`156c661a45a4`, 3 rendered views. **320 written, 309 scorable** — 11 were
infrastructure outages that never reached the model and are excluded from every
denominator.

| metric | scope | value |
|---|---|---|
| Goal coverage | feasible | **0.580** (775/1182 goals) |
| Functional assignment coverage | feasible | **0.589** (1192/1492 slots = 79.9%) |
| End-to-end | feasible | **80/192 = 42%** any order · 47/192 = 24% exact order |
| Correct rejection | infeasible | 65/117 = 56% |
| False completion | infeasible | 52/117 = 44%, of which **1** claimed completion |
| Whole benchmark, strict | all | **145/309 = 47%** |
| Regions inspected | feasible 1.90 · infeasible 2.67 | kitchen 3.99 · living_room 0.00 · workshop 2.32 |

| domain | trials | feasible | goal cov. | FAC | end-to-end | rejected |
|---|---:|---:|---:|---:|---:|---:|
| kitchen | 113 | 54 | 0.787 | 0.732 | 33/54 = 61% | 34/59 = 58% |
| living_room | 99 | 60 | 0.487 | 0.563 | 19/60 = 32% | 27/39 = 69% |
| workshop | 97 | 78 | 0.509 | 0.511 | 28/78 = 36% | 4/19 = 21% |

Two findings worth stating explicitly. **Infeasible trials inspect more regions
than feasible ones** (2.67 against 1.90): when the required object does not
exist, the search exhausts the scene before giving up. And matching the GT
action multiset is **sufficient but not necessary** — all 80 runs that match the
actions satisfy every goal, while 3 satisfy every goal by a different sequence.

> **Cite `rescored_current_scorer.json`, not `final_10x32.json`.** The latter is
> the original pass, kept for provenance, and carries pre-fix workshop FAC
> (0.000) and pre-fix goal coverage. Four scorer defects were found and fixed
> offline — all in scoring, none in the pipeline, no trial re-run. The largest:
> the identity resolver covered kitchen only, so workshop FAC read 0% instead of
> 68.7%; and Living Room's GT asked for `cup` and `saucer` as separate roles
> when the detector emits `cup_or_saucer`, making those slots unsatisfiable by
> construction and holding LR goal coverage at 0.087 instead of 0.487.

## 2. Failure analysis

`experiments/zs_pipeline_arms/scored/failure_counts.{json,txt}` ·
chart: https://claude.ai/code/artifact/4f73c5b7-cde2-4be1-93a3-8cc2a02676e2

First-cause attribution over the 192 feasible trials — each failing trial is
assigned to the **earliest** stage that broke, which is why the later stages
look small. Counts are over trials, not over failures, so the column sums to
the trial total rather than to 100%.

| stage | kitchen | living_room | workshop | overall |
|---|---:|---:|---:|---:|
| Task Specification | 1/54 (1.9%) | 8/60 (13.3%) | 10/78 (12.8%) | 19/192 (9.9%) |
| **Graph Compilation** | 20/54 (37.0%) | 29/60 (48.3%) | 22/78 (28.2%) | **71/192 (37.0%)** |
| Object Discovery | 0 | 0 | 15/78 (19.2%) | 15/192 (7.8%) |
| Functional Assignment | 0 | 3/60 (5.0%) | 0 | 3/192 (1.6%) |
| Planning | 0 | 1/60 (1.7%) | 0 | 1/192 (0.5%) |
| **Success** | 33/54 (61.1%) | 19/60 (31.7%) | 31/78 (39.7%) | **83/192 (43.2%)** |

**Graph compilation is the dominant blocker** — turning the FM's own words into
an executable graph, not producing them and not planning over them. Its two
largest sub-causes are role collisions (`AMBIGUOUS_ROLE_MAPPING`, 75) and
unmapped operations. That is what motivated everything in sections 4 and 5.

## 3. Inspection-order ablation

`benchmark_reports/order_fm,_random,_worst/` (309 trials each) ·
[full write-up](benchmark_reports/INSPECTION_ORDER_ABLATION/README.md)

Three arms replaying the **same 309 frozen FM responses**, differing only in
region inspection order. Paired trial by trial, so FM sampling is not a
confound.

| arm | regions/trial | vs FM (paired) | workshop opening | vs FM |
|---|---:|---:|---:|---:|
| FM-ranked | 2.16 | — | **240.7 s** | — |
| Random | 2.14 | −0.02 ±0.06, p = 0.460 | 233.1 s | −7.6 s |
| Worst-case (privileged) | 2.39 | **+0.23 ±0.08, p < 0.001** | 261.3 s | **+20.6 s** |

**The FM ranking beats an adversary and ties a coin.** Random is
indistinguishable; the adversarial order — empty regions first, built from
ground truth the pipeline never sees — is reliably worse.

Two things the write-up establishes that the table cannot:

- **Physical opening dominates, and it was being left out.** A workshop trial
  spends ~27 s grounding and ~240 s driving drawers and a cabinet open, measured
  with contact-gated robot actuation from the home position. The worst-case
  penalty is +20.6 s of actuation against +3.6 s of grounding — reporting this
  in grounding seconds understates it roughly six-fold.
- **The timing columns are drift, not effect.** Living Room declares no
  inspectable regions, so all three arms run it identically, and it still
  differs by +0.17 s (random) and +1.23 s / 18.3% (worst). That makes it a
  negative control, and it is why the worst-case grounding time is not reported
  as a measurement. The opening cost is drift-free by construction — measured
  per-region cost times which regions opened, no wall clock involved.

**309/309 trials reach the same terminal status in all three arms.** A cheaper
ordering that silently changed answers would be a defect; the data rules it out.

Why FM-vs-random finds nothing: **152 of 210 orderable pairs (72%) cannot differ
by construction** — 126 exhaust every region in both arms and 26 inspect none.
Kitchen inspects 3.91 of 5 regions and workshop 2.32 of 3. A benchmark whose
searches are near-exhaustive cannot discriminate ordering policies however good
the ranking is. The worst-case arm separates only because it is allowed to
cheat.

## 4. Alias resolution — can zero-shot NLI do canonicalization at all?

`experiments/alias_resolution_study/` ·
[study README](experiments/alias_resolution_study/README.md) ·
metrics in `results/aggregate_metrics.json` and `results/layer_metrics.json`

Offline study, `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` rev `b2730f16`,
CPU. Ran under strict isolation: no pipeline file edited, no FM call made, no
frozen output touched — proven by before/after SHA-256 in
`results/isolation_audit.json`, which records `identical: true` for every
pre-existing file.

**The relation layer, and the finding that made the rest viable** (n = 119
reference-resolved instances, top-1 agreement):

| condition | instance | unique phrase | what the model is shown |
|---|---:|---:|---|
| ZS-Global | 0.370 | 0.387 | the phrase against all 9 predicates |
| ZS-Typed | 0.555 | 0.566 | phrase + the argument types |
| **ZS-Context** | **0.866** | **0.868** | phrase + argument types + endpoint identities |

**Endpoint context is what makes zero-shot canonicalization work** — 0.370 to
0.866 on identical phrases and an identical label set. Living Room moves from
0.032 to 0.710. The phrase alone is not enough information to pick a predicate;
what the predicate connects is.

**All four vocabularies** (`results/layer_metrics.json`), against a chance
baseline computed per layer:

| layer | instances | top-1 | chance | lift | weakest domain |
|---|---:|---:|---:|---:|---|
| region | 485 | **0.893** | 0.259 | 3.45× | workshop 0.785 |
| capability | 460 | **0.880** | 0.417 | 2.11× | living_room 0.831 |
| role | 1152 | **0.810** | 0.202 | 4.02× | **workshop 0.411** |

Workshop role resolution is the clear weak point, and the confusion matrix says
why: `driver → fastener` 72 times and `repair_target → fastener` 69. The
vocabulary asks the model to distinguish the tool, the workpiece and the
hardware, and the phrasing frequently does not.

## 5. Zero-shot canonicalization in the pipeline

`benchmark_reports/zs_arm,_zs4_top1,_zs4_thresh/` ·
`experiments/zs_pipeline_arms/` ·
[arms README](experiments/zs_pipeline_arms/README.md)

The fallback put into the pipeline as a shadow and the frozen 320 replayed
through it. Augment-only: the shipped resolver is consulted first and the
zero-shot model runs only where it returns nothing, so a trial the current
system already handles takes the identical path and the arm cannot regress it.

| stage | baseline | with ZS | change |
|---|---:|---:|---|
| Task Specification | 19/192 (9.9%) | 19/192 (9.9%) | — |
| Graph Compilation | 71/192 (37.0%) | 66/192 (34.4%) | −5 |
| **Object Discovery** | 15/192 (7.8%) | **2/192 (1.0%)** | **−13** |
| Functional Assignment | 3/192 (1.6%) | 3/192 (1.6%) | — |
| Planning | 1/192 (0.5%) | 2/192 (1.0%) | +1 |
| **Success** | 83/192 (43.2%) | **100/192 (52.1%)** | **+17** |

**+17 successes, 43.2% → 52.1%, and almost all of it is workshop** — 31/78
(39.7%) to 48/78 (61.5%). The mechanism is Object Discovery collapsing from 15
to 2: better region canonicalization sends the robot to the right drawer, so
objects that were previously never found now are. Kitchen and Living Room are
unchanged at 33/54 and 19/60.

**Not yet settled: which configuration ships.** Three layers at top-1 and four
layers at margin ≥ 0.15 are both defensible, and `zs4_thresh` was stopped at 22
of 309, so that comparison is incomplete.

**Consequence for section 3.** The fallback changes the search contract —
`region_ranking` in 109 trials, `region_order_used` in 81, `inspected_regions`
in 84. If it ships, the inspection-order ablation has to be re-run, because its
dependent variable is exactly what this moves.

## 6. Evidence ablations

`runs/gt_evidence_ablation/`, `runs/fm_evidence_ablation/` ·
shadows `mujoco_scenes/fm_evidence_ablation.py`, `fm_ablation_shadow.py` ·
[FM_EVIDENCE_ABLATION.md](FM_EVIDENCE_ABLATION.md)

Withholds semantic, unary and binary evidence at grounding to establish which
evidence classes the grounding actually needs.

## 7. Known open problems

Carried in full in [PIPELINE.md §6](PIPELINE.md).

**Role collisions.** `AMBIGUOUS_ROLE_MAPPING`, 75 occurrences: two FM roles
resolving to the same canonical role. An augment-only fallback can never fire
for it, because the mapper already returned an answer — the answer is the
problem. This is the largest remaining sub-cause of the dominant failure stage.

**One FM role, two canonical roles.** Living Room's characteristic failure. The
model writes a single `placement_surface` whose own stated function covers both
the refreshments and the remote; it canonicalizes to
`PERSONAL_CUP_SAUCER_REGION`, and `SUPPORT_ENTERTAINMENT_CONTROL` then finds no
capability whose signature accepts it, so the remote placement is dropped — 10
of 60 feasible Living Room trials, all of which place the drinkware correctly
and stop two actions short. Quantitatively: Living Room averages 2.05 region
roles against kitchen's 1.09, and 19/19 against 1/11 on both-canonical
resolution.

This is a design change, not a patch: role resolution would have to become
**operation-relative** rather than one canonical name per FM role.
`mujoco_scenes/fm_operation_relative_region_shadow.py` probes it and **does not
work** — it guards on the target slot being a region, but the capability carries
the region as its *source*, so it returns early and reproduces the baseline
exactly. It is committed unrun with the diagnosis in its docstring.

**`NO_GLOBAL_REGION_ASSIGNMENT`.** 8 feasible Living Room trials are declared
infeasible with a message that says what failed but not which constraint could
not be met.

## 8. Reading this repository honestly

**What is not in git.** `benchmark_reports/` holds 60 GB locally and ~700 MB
here. The measurement record is committed — raw FM responses, sanitized and
compiled specifications, G_F, run manifests with the frozen search contract,
grounding results, symbolic problems, validated results, every scoring pass.
Rendered views, point clouds and per-stage perception dumps are not; they are
regenerable with `--spec-source raw-replay`, which makes no model calls. See
[benchmark_reports/README.md](benchmark_reports/README.md).

**Discarded runs are kept on purpose.** Directories prefixed `_discarded_` or
`_aborted_` each record a run that had to be thrown away — a dirty working tree
during a live benchmark, swap thrashing that contaminated timing, a partial arm.
The reason is worth more than the disk space.

**Every ablation is a shadow.** No pipeline file is edited by any arm on this
branch. A context manager patches named functions in process and restores them
on exit. The pattern and the rule that keeps it honest — *verify the patch point
on one trial before running 309* — are in [PIPELINE.md §5](PIPELINE.md). That
rule is there because it was learned the hard way twice: `_map_role` is
overridden by `resolve_role_type_hypotheses` and fired 537 times while changing
nothing, and `_participant_slot_fit` is never reached because
`interpret_operation` rejects earlier on endpoint signature. Both produced a
convincing null result and both were visible in a single trial.

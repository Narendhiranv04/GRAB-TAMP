# Inspection-Order Ablation — FM-ranked vs Random vs Worst-Case

Does the FM's region ranking reduce what inspection costs?

**Against a random order, no — and the benchmark mostly cannot tell. Against an
adversarial order, yes, and the cost that moves is robot actuation, not
perception.** Both conclusions come from the same three arms, and neither
changes a single answer: all 309 trials reach the same terminal status in every
arm.

## Design

All three arms replay the **same 309 frozen FM responses** from the 320-trial
run. Scene, variant, functional requirement graph, model, sampler and commit
are identical within every pair; only the order regions are inspected differs.

| arm | inspection order |
|---|---|
| FM-ranked | the FM's own ranking, system-completed |
| Random | seeded Fisher–Yates shuffle of the region universe, reseeded per trial |
| Worst-case | **privileged**: empty regions first, then ascending content count |

The worst-case order is built from `container_contents` and
`privileged_get_storage_contents` — ground truth the pipeline never sees. It is
an upper bound on what bad ordering can cost, not a baseline any policy could
produce. Its purpose is to bound the other end of the range the FM ranking sits
in, since random alone leaves the range unmeasured.

Replay starts from `fm_diagnostics/fm_call_001.json`, not from the canonicalized
specification. Canonicalization is where region proposals become canonical ids,
so replaying a specification written before that mapping was corrected would
carry its loss forward. The FM is not called again — its answer is read from
disk, which is what holds the specification identical across arms. A fresh call
at temperature 0.6 would return a different specification, and differences
between specifications would then be indistinguishable from differences caused
by ordering. Everything downstream still runs for real: vocabulary,
open-vocabulary semantics, point-cloud geometry, G_O, region search, grounding
and A*.

`region_ranking` reaches exactly one decision point in the pipeline
(`search_contract.py`, where the inspection order is built) and touches nothing
else, so replacing it is a complete ablation of it.

**Orders verified, not assumed.** Of 210 orderable pairs, the random arm drew
the random policy in 210 and received a genuinely different order in 171; the
worst-case arm differs from the FM order in 167. The remainder are orders that
coincided with the FM's, expected when workshop has only six permutations.

## Regions inspected

| arm | kitchen (of 5) | living_room | workshop (of 3) | overall |
|---|---:|---:|---:|---:|
| FM-ranked | 3.91 | 0.00 | 2.32 | **2.16** |
| Random | 3.88 | 0.00 | 2.28 | 2.14 |
| Worst-case | 4.33 | 0.00 | 2.58 | 2.39 |

Paired differences against the FM arm, Wilcoxon signed-rank, two-sided:

| comparison | kitchen | workshop | overall | p (overall) |
|---|---:|---:|---:|---:|
| Random − FM | −0.03 ±0.14 | −0.04 ±0.10 | −0.02 ±0.06 | 0.460 |
| Worst − FM | **+0.42 ±0.18** | **+0.26 ±0.13** | **+0.23 ±0.08** | **0.000** |

Random is indistinguishable from the FM ranking. The adversarial order is
reliably worse. The FM ranking is therefore doing something — it is just not
doing anything a shuffle does not also do on this benchmark.

## Physical opening cost — the term that actually dominates

Grounding time measures perception and φ. It does not measure the robot opening
anything, and the opening is an order of magnitude larger.

Measured per region, contact-gated robot actuation from the home position,
verified against ground truth (`workshop_open_costs.json`):

| region | opening time |
|---|---:|
| `TOOL_CABINET` | 141.6 s |
| `RIGHT_DRAWER` | 82.6 s |
| `LEFT_DRAWER` | 81.6 s |

Composed over the regions each trial actually inspected
(`scripts/score_inspection_open_cost.py`):

| arm | workshop opening | vs FM | workshop grounding | vs FM |
|---|---:|---:|---:|---:|
| FM-ranked | **240.7 s** | — | 27.5 s | — |
| Random | 233.1 s | −7.6 s | 27.2 s | −0.3 s |
| Worst-case | 261.3 s | **+20.6 s** | 31.2 s | +3.6 s |

Two things follow. First, the worst-case penalty is **+20.6 s of actuation
against +3.6 s of grounding** — reporting this ablation in grounding seconds
understates the effect roughly six-fold. Second, the actuation figure is not
drift-contaminated: it is measured per-region cost times which regions were
opened, so it contains no wall-clock measurement at all, unlike the grounding
column below.

Random remains cheaper than the FM ranking (−7.6 s), consistent with the null
on regions inspected. The honest summary is that the FM ranking beats an
adversary and ties a coin.

**Kitchen and Living Room have no actuation cost to report, and that is a
property of the pipeline, not a missing measurement.**
`run._get_exploration_actuation` returns `direct_sim_articulation` for kitchen
*unconditionally* — kitchen containers open by setting the simulator joint with
or without `--dry-run` — and `not_applicable` for living_room. Only workshop has
a robot-actuated opening path. Kitchen's +0.42 region penalty under the
adversarial order is consequently the larger effect in region terms and the one
with no physical cost attached, which is worth stating rather than quietly
averaging away.

## Why FM-vs-random finds nothing

Per-pair outcome across 210 orderable pairs (kitchen + workshop), random vs FM:

| outcome | pairs | |
|---|---:|---|
| tie | 170 | 81% |
| random inspected fewer | 24 | 11% |
| random inspected more | 16 | 8% |

24 against 16 is coin-flipping. And of the 170 ties, **126 exhausted every
region in both arms** and **26 inspected nothing in both arms** — so **152 of
210 pairs (72%) cannot differ by construction.** When both arms open all five
kitchen drawers, the order is a permutation of identical work. Those pairs
dominate the means and compress them toward zero. Restricted to the 58 pairs
where ordering could matter: FM 2.53 regions against random 2.41.

Kitchen inspects 3.91 of 5 regions and workshop 2.32 of 3. An ordering policy
can only pay in proportion to the search it avoids, and these tasks need objects
that are usually hidden, so the search rarely stops early. **This is a property
of the benchmark, not of the ranking.** A benchmark whose searches are
near-exhaustive cannot discriminate inspection-order policies however good the
ranking is. The worst-case arm separates precisely because it is allowed to
cheat: it knows which regions are empty and opens them first, manufacturing the
early-stop opportunities the benchmark otherwise does not contain.

## Grounding time in the timing columns is measurement drift

Living Room declares no inspectable regions, so the contract validator refuses a
reordering for it and **all three arms run it identically** — same policy, no
seed, zero regions. Their Living Room cells are the same computation.

They still differ: **+0.17 s in the random arm (p = 0.001) and +1.23 s, 18.3%,
in the worst-case arm.**

That is not an ordering effect, because there is no ordering to differ. It is
drift between arms run back to back on one machine, and it makes Living Room a
negative control that sets the floor below which a timing difference means
nothing.

| arm | scene | observed | − additive drift | − relative drift |
|---|---|---:|---:|---:|
| Random | kitchen | +0.50 s | +0.33 s | −0.12 s |
| | workshop | −0.31 s | −0.47 s | −0.99 s |
| | overall | +0.14 s | −0.03 s | −0.35 s |
| Worst | kitchen | +6.36 s | +5.12 s | +1.81 s |
| | workshop | +3.64 s | +2.40 s | −1.40 s |
| | overall | +3.87 s | +2.63 s | +0.23 s |

For the random arm the difference collapses to roughly zero, and workshop is
already negative before any correction. For the worst-case arm an 18.3% control
drift is large enough to consume most of the observed difference, so **the
worst-case grounding time is not reported as a measurement.** Its
regions-inspected and opening-cost results carry the claim instead; neither
depends on wall-clock timing.

Planning time is 0.008 s against 0.008 s against 0.009 s and never exceeds
0.024 s in any cell. Ordering moves perception and actuation cost, not search
cost.

## Correctness is unaffected

**309 / 309 trials reach the same terminal status in all three arms**, in every
scene. This matters more than the cost result: a cheaper ordering that silently
changed answers would be a defect, and an adversarial ordering that broke
answers would mean the search was not exhaustive. The data rules out both.

## An earlier version of this ablation was invalid

The first run produced the same null from a broken premise. Region proposals
were being lost in canonicalization, so the FM arm ranked only **1.10 of 5**
kitchen regions and the deterministic system completion supplied the rest in
canonical order. That measured canonical order against random order while
reporting it as FM against random.

The cause was in `KITCHEN_REGION_ALIASES`: geometrically false entries (C1 as a
"base cabinet"; D1 as "top" and D2 as "bottom" when `kitchen_base.xml` puts both
at z = 0.46) and side-less aliases that sent every generic "wall cabinet" to C2
— 107 against C1's 39. After grounding the aliases in the scene geometry the FM
arm ranks **2.43 of 5**, trials with no ranking at all fell from 37 to 8, and
the per-region counts came out symmetric (C1 83 = C2 83, D1 36 = D2 36, B1 89).
The null survives, now with the FM's ranking actually exercised. Terminal
statuses were unchanged by the fix across all 275 comparable trials.

## Reproducing

```bash
python3 scripts/run_search_order_replay_timing.py \
    --output-root benchmark_reports/order_fm     --order-mode auto
python3 scripts/run_search_order_replay_timing.py \
    --output-root benchmark_reports/order_random --order-mode random
python3 scripts/run_search_order_replay_timing.py \
    --output-root benchmark_reports/order_worst  --order-mode worst

python3 scripts/score_inspection_order_ablation.py \
    --fm     benchmark_reports/order_fm \
    --random benchmark_reports/order_random \
    --out    benchmark_reports/INSPECTION_ORDER_ABLATION/inspection_order_ablation.json
python3 scripts/score_inspection_order_ablation.py \
    --fm     benchmark_reports/order_fm \
    --random benchmark_reports/order_worst \
    --out    benchmark_reports/INSPECTION_ORDER_ABLATION/fm_vs_worst.json
python3 scripts/score_inspection_open_cost.py \
    --arms order_fm order_random order_worst \
    --out  benchmark_reports/INSPECTION_ORDER_ABLATION/open_cost.json
```

All arms must run on one machine, back to back, with nothing else competing:
grounding time is only interpretable within a machine, and the Living Room
control shows that even back-to-back runs drift by 2.5% to 18%. One earlier
attempt ran a replay pass concurrently, drove the machine into swap at 11 MB/s,
and its 77 trials had to be discarded as timing-contaminated.

Coverage is 309 of the original 320 trials. The 11 absent are infrastructure
outages that never reached the model, so they are not observations of the
method.

## Outstanding

These three arms predate the zero-shot canonicalization change, which alters
`region_ranking` in 109 trials, `region_order_used` in 81 and
`inspected_regions` in 84. **If zero-shot canonicalization ships, this ablation
has to be re-run** — its dependent variable is exactly what that change moves.

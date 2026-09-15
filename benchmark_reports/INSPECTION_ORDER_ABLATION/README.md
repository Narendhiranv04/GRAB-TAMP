# Inspection-Order Ablation — FM-ranked vs Random

Does the FM's region ranking reduce inspection cost against a random order?

**No, and the benchmark mostly cannot tell.** 72% of orderable trials either
exhaust every region or inspect none in both arms, so ordering is incapable of
changing their cost. Across the remainder the two arms are indistinguishable,
and every trial reaches the same conclusion under both.

## Design

Both arms replay the **same 309 frozen FM responses** from the 320-trial run.
Scene, variant, functional requirement graph, model, sampler and commit are
identical within every pair; only the order regions are inspected differs.

| arm | inspection order |
|---|---|
| FM-ranked | the FM's own ranking, system-completed |
| Random | seeded Fisher-Yates shuffle of the region universe, reseeded per trial |

Replay starts from `fm_diagnostics/fm_call_001.json`, not from the canonicalized
specification. Canonicalization is where region proposals become canonical ids,
so replaying a specification written before that mapping was corrected would
carry its loss forward. The FM is not called again -- its answer is read from
disk, which is what holds the specification identical across arms. A fresh call
at temperature 0.6 would return a different specification, and differences
between specifications would then be indistinguishable from differences caused
by ordering. Everything downstream still runs for real: vocabulary,
open-vocabulary semantics, point-cloud geometry, G_O, region search, grounding
and A*.

`region_ranking` reaches exactly one decision point in the pipeline
(`search_contract.py`, where the inspection order is built) and touches nothing
else, so replacing it with a shuffle is a complete ablation of it.

**Shuffle verified, not assumed**: of 210 orderable pairs, 210 drew the random
policy and 171 received an order genuinely different from the FM's. The other 39
are shuffles that coincided with the FM order, expected when workshop has only
six permutations.

## The result that matters

Per-pair outcome across 210 orderable pairs (kitchen + workshop):

| outcome | pairs | |
|---|---:|---|
| tie | 170 | 81% |
| random inspected fewer | 24 | 11% |
| random inspected more | 16 | 8% |

24 against 16 is coin-flipping. And of the 170 ties:

- **126 exhausted every region in both arms**
- **26 inspected nothing in both arms**

So **152 of 210 pairs (72%) cannot differ by construction.** When both arms open
all five kitchen drawers, the order is a permutation of identical work. Those
pairs dominate the means and compress them toward zero.

Restricted to the 58 pairs where ordering could matter: FM 2.53 regions against
random 2.41.

## Paired means

Random minus FM. Wilcoxon signed-rank, two-sided.

| scene | n | metric | FM-ranked | Random | diff | 95% CI | p |
|---|---:|---|---:|---:|---:|---:|---:|
| kitchen | 113 | Regions inspected | 3.91 | 3.88 | −0.03 | ±0.14 | 0.670 |
| | | Candidate checks | 134.6 | 134.6 | 0.0 | ±0.0 | — |
| | | Grounding time (s) | 24.85 | 25.35 | +0.50 | ±0.63 | 0.039 |
| living_room | 99 | Regions inspected | 0.00 | 0.00 | 0.00 | ±0.00 | — |
| | | Candidate checks | 23.2 | 23.2 | 0.0 | ±0.0 | — |
| | | Grounding time (s) | 6.77 | 6.94 | +0.17 | ±0.09 | 0.001 |
| workshop | 97 | Regions inspected | 2.32 | 2.28 | −0.04 | ±0.10 | 0.492 |
| | | Candidate checks | 13.4 | 13.3 | −0.2 | ±0.5 | 0.538 |
| | | Grounding time (s) | 27.52 | 27.22 | −0.31 | ±1.01 | 0.393 |
| **overall** | **309** | **Regions inspected** | **2.16** | **2.14** | **−0.02** | **±0.06** | **0.460** |
| | | **Candidate checks** | **60.9** | **60.8** | **−0.0** | **±0.1** | **0.538** |
| | | **Grounding time (s)** | **19.90** | **20.04** | **+0.14** | **±0.39** | **0.054** |

The overall region difference is −0.02 with a confidence interval of ±0.06. Do
not read the direction of a mean that sits well inside its own interval; the
per-pair table above is what the data supports.

Planning time is 0.008 s against 0.008 s and never exceeds 0.021 s in any cell.
Ordering moves perception cost, not search cost, and here it moves neither.

## The grounding-time difference is measurement drift

Living Room declares no inspectable regions, so the contract validator refuses a
random order for it and **both arms run it identically** -- same policy, no seed,
zero regions. Its two arms are the same computation.

It still differs by **+0.17 s at p = 0.001**.

That is not an ordering effect, because there is no ordering to differ. It is
drift between arms run back to back on one machine. Living Room is therefore a
negative control and sets the floor below which a timing difference means
nothing.

| scene | observed | minus additive drift (+0.167 s) | minus relative drift (+2.47%) |
|---|---:|---:|---:|
| kitchen | +0.50 s | +0.33 s | −0.12 s |
| workshop | −0.31 s | −0.47 s | −0.99 s |
| overall | +0.14 s | −0.03 s | −0.35 s |

Workshop is already negative before any correction, and the overall figure
collapses to roughly zero. **Report the raw numbers with this control stated. Do
not report "random ordering is slower" -- the control refutes it.**

## Why there is nothing to find

Kitchen inspects **3.91 of 5** regions and workshop **2.32 of 3**. An ordering
policy can only pay in proportion to the search it avoids, and these tasks need
objects that are usually hidden, so the search rarely stops early.

This is a property of the benchmark, not of the ranking. A benchmark whose
searches are near-exhaustive cannot discriminate inspection-order policies
however good the ranking is.

## Correctness is unaffected

**309 / 309 trials reach the same terminal status in both arms**, in every
scene. This matters more than the cost result: a cheaper ordering that silently
changed answers would be a defect, and the data rules that out.

## An earlier version of this ablation was invalid

The first run of this experiment produced the same null from a broken premise.
Region proposals were being lost in canonicalization, so the FM arm ranked only
**1.10 of 5** kitchen regions and the deterministic system completion supplied
the rest in canonical order. That measured canonical order against random order
while reporting it as FM against random.

After the region aliases were grounded in scene geometry, the FM arm ranks
**2.43 of 5** and trials with no ranking at all fell from 37 to 8. The null
survives, now with the FM's ranking actually exercised. Terminal statuses were
unchanged by the fix across all 275 comparable trials.

## Reproducing

```
python3 scripts/run_search_order_replay_timing.py \
    --output-root benchmark_reports/order_fm     --order-mode auto
python3 scripts/run_search_order_replay_timing.py \
    --output-root benchmark_reports/order_random --order-mode random
python3 scripts/score_inspection_order_ablation.py \
    --fm     benchmark_reports/order_fm \
    --random benchmark_reports/order_random \
    --out    benchmark_reports/INSPECTION_ORDER_ABLATION/inspection_order_ablation.json
```

Both arms must run on one machine, back to back, with nothing else competing:
grounding time is only interpretable within a machine, and the Living Room
control shows that even back-to-back runs drift by about 2.5%.

Coverage is 309 of the original 320 trials. The 11 absent are infrastructure
outages that never reached the model, so they are not observations of the
method.

# Zero-shot canonicalization as a pipeline arm

The offline study in `../alias_resolution_study/` asked whether a zero-shot
NLI model can map free-form FM wording onto the fixed vocabularies at all.
This directory holds what happened when that fallback was put **into** the
pipeline and the frozen 320 was replayed through it.

## What is here

| path | what it is |
|---|---|
| `scored/baseline_scored.json` | the 309 scorable trials with the shipped resolvers only |
| `scored/zs_scored.json` | same trials, zero-shot fallback on capability, relation and region |
| `scored/zs4_top1_scored.json` | same, plus the role layer at top-1 |
| `scored/verif_ablation.json` | independent-validation ablation over the same trials |
| `scored/failure_counts.{json,txt}` | first-cause failure attribution behind the failure-stage table |
| `runners/` | the shell drivers actually used, kept so the arms can be reproduced verbatim |

`runners/server_run.sh` targets the ngrok-forwarded GPU box and
`runners/laptop_zs.sh` runs the same work on CPU locally. Only the offline
NLI model differs in placement — neither makes FM calls, so both read the
archived responses and produce the same arms.

## Reading the numbers

Every arm is a **paired replay**: the same 309 archived FM responses, the
same variants, differing only in whether the zero-shot fallback is
consulted when the shipped resolver returns nothing. Differences are
therefore attributable to canonicalization and nothing else — no FM
sampling, no perception variation, no re-planning noise.

The fallback is augment-only. It cannot regress a trial the current system
already resolves, because it never runs on one. What it can and does change
is the **search contract**: `region_ranking` in 109 trials,
`region_order_used` in 81, `inspected_regions` in 84. Any ablation whose
dependent variable is region inspection has to be re-run under whichever
configuration ships, and the inspection-order and verification ablations
both qualify.

## What is not settled

Which configuration ships. Three layers at top-1 and four layers at
margin >= 0.15 are both defensible, and `zs4_thresh` — the margin-gated
four-layer arm — was stopped at 22 of 309, so the comparison is incomplete.
The margin threshold itself came out of the accuracy-coverage curve in the
offline study, not from these arms.

The role layer does not address `AMBIGUOUS_ROLE_MAPPING` (75 occurrences),
which is two FM roles colliding on one canonical role. An augment-only
fallback never fires there: the mapper already returned an answer, and the
answer is the problem.

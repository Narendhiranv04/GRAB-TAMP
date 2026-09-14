# Can a zero-shot classifier replace hand-written relation aliases?

**No, not from the relation phrase alone.** Excluding rows where the type filter
left a single candidate and the answer was therefore correct by construction,
the zero-shot classifier scores **0.376 against a 0.419 chance baseline — a
lift of 0.90x, slightly below chance.** It succeeds only in Kitchen, where the
choice is between two semantically distant predicates.

The reason is more useful than the number: the frozen mapping is **not purely
lexical**. The same phrase maps to different predicates depending on which roles
it connects, and a classifier shown only the phrase cannot recover that.

## Motivation

The FM invents its own relation vocabulary while execution needs a fixed
predicate interface. Across the frozen 320-trial run it produced **973 relation
instances in 510 distinct phrasings** -- `contains`/`CONTAINS`/`Contains`/
`contained_in`/`holds`/`filled_with`, `placed_on`/`is_placed_on`/`located_on`/
`rests_on`/`on`. Per predicate the variation is near-total:

| canonical predicate | instances | distinct phrasings |
|---|---:|---:|
| COMPATIBLE_WITH | 34 | 31 |
| INSERTABLE_IN | 28 | 26 |
| COMPATIBLE_WITH_TARGET | 23 | 22 |
| FITS_SET_ON | 21 | 17 |
| FITS_ON | 7 | 7 |

Almost every occurrence is worded differently, so enumerating aliases by hand is
a losing race. That is the motivation; whether an off-the-shelf classifier wins
it is the question.

## Setup

Frozen final 320-trial run (`benchmark_reports/full320_s01..s03`, scored in
`FINAL_10x32_RESULTS/final_10x32.json`), 309 archived FM responses; 21 trials
were infrastructure outages that never reached the model. **No FM calls were
made.** The pipeline was not modified.

The manual baseline is the pipeline's own `relation_interpreter`, imported
read-only at commit `a977544d`, so the baseline is the code that actually ran.

Classifier: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` (revision
`b2730f16`), zero-shot, no fine-tuning, CPU, transformers 4.57.3, torch 2.9.1.
Candidate hypotheses are built mechanically as `NAME: registry description`.
The manual cue table was never shown to the classifier, as prompt, example or
label text -- showing it would make the study circular.

Two conditions: **ZS-Global** (all active binary predicates in the domain) and
**ZS-Typed** (further restricted to those whose registry signature admits the
declared entity kinds).

## Evaluation sets

`REFERENCE_RESOLVED` -- 119 instances / 106 unique phrases that the manual
resolver mapped to an active binary executable predicate. The manual resolver
*defines* these labels, so its score here is 1.0 by construction and is not an
independent evaluation. Reported for it: coverage.

`FROZEN_UNRESOLVED` -- 583 instances the manual resolver could not map. **No
accuracy is computed for these**, since calling them errors would assume the
manual cues were complete, which is the assumption under test.

## Result

| resolver | hand aliases | training | unique-phrase coverage | top-1 vs frozen | chance | lift |
|---|---|---|---:|---:|---:|---:|
| Frozen lexical (baseline) | Yes | No | **47.2%** (252/534) | 1.00 † | — | — |
| Zero-shot NLI, ZS-Global | No | No | 100% ‡ | 0.370 | 0.351 | 1.05x |
| Zero-shot NLI, ZS-Typed | No | No | 100% ‡ | 0.555 | 0.585 | 0.95x |
| Zero-shot NLI, ZS-Typed, forced-choice rows removed | No | No | 100% ‡ | **0.376** | **0.419** | **0.90x** |

† Not an independent evaluation: the frozen resolver supplies the reference labels.
‡ The classifier always emits a label, so "coverage" is not comparable to the
baseline's; it abstains only under a threshold, analysed separately.

**The ZS-Typed gain over ZS-Global is not a gain.** Type filtering shrinks the
candidate set, so chance rises from 0.351 to 0.585 — faster than accuracy does.
34 of the 119 rows collapse to a single candidate and are correct by
construction; removing them leaves 0.376 against 0.419 chance.

### Per domain (ZS-Typed, forced-choice rows removed)

| domain | n | top-1 | chance | lift |
|---|---:|---:|---:|---:|
| kitchen | 28 | **0.893** | 0.500 | **1.79x** |
| living_room | 31 | 0.065 | 0.282 | 0.23x |
| workshop | 26 | 0.192 | 0.494 | 0.39x |

Kitchen is the only genuine success, and it is the easy case: a two-way choice
between `INSERTABLE_IN` ("cross-section fits inside the opening") and
`REACHES_BOTTOM` ("length reaches the bottom"), which are semantically far
apart. Where predicates are close, the classifier is worse than guessing.

## Why it fails

| gold | predicted | n |
|---|---|---:|
| COMPATIBLE_WITH_TARGET | REACHES_TARGET | 21 |
| FITS_SET_ON | ACCESSIBLE_FROM_BOTH_SEATS | 19 |
| FITS_ON | ACCESSIBLE_FROM_BOTH_SEATS | 7 |
| INSERTABLE_IN | REACHES_BOTTOM | 3 |

The decisive example is the phrase `placed_on`, whose frozen label is
`FITS_SET_ON` -- "region surface area and boundary support the composite cup and
saucer set". Nothing in the words "placed on" says *fits*. The manual resolver
reaches that predicate from the **endpoints**: a cup-and-saucer set placed on a
surface induces a geometric fit check. The same phrase with a different subject
maps to `FITS_ON`.

So the real task is not

```
phrase -> predicate
```

but

```
(phrase, subject role, object role) -> predicate
```

The classifier in this study sees only the phrase. Entity kinds were used to
*filter* candidates but never entered the premise it scored. That is a genuine
limitation of this implementation, not only of zero-shot classification, and it
is the obvious next experiment: put the role identities into the premise.

A second, distinct failure is `ACCESSIBLE_FROM_BOTH_SEATS` absorbing 28
predictions while appearing **zero** times in the gold labels. Its registry
description is the most generic sentence in the Living Room set, so it attracts
any placement-flavoured phrase. The predictions collapse onto whichever
description reads most broadly, which is a property of using registry prose as
label text.

## Frozen unresolved relations

583 instances (of which the manual resolver mapped none). Confidence buckets
under ZS-Typed:

| bucket | n |
|---|---:|
| AMBIGUOUS | 313 |
| LOW_CONFIDENCE | 198 |
| HIGH_CONFIDENCE_PROPOSAL | 72 |

**These are confidence bands, not correctness.** No reference label exists, and
given the classifier scores below chance where labels *do* exist, the 72
high-confidence proposals should not be treated as recovered relations without
independent verification.

## What this does and does not show

It shows that hand-enumerated aliases cover **47.2%** of the distinct phrasings
the FM actually produced, and that an off-the-shelf zero-shot classifier does
**not** close that gap from the phrase alone.

It does not show that the approach is hopeless. The endpoint-conditioned framing
is untested here, and Kitchen's 1.79x lift shows the model can separate
predicates when their descriptions are genuinely distinct.

It says nothing about real-robot deployment. This studies the linguistic
interface-normalization layer only. Physical implementations remain
embodiment-specific: `INSERTABLE_IN` still needs a simulator checker in
simulation and a perception/geometric checker on hardware.

## Caveat on scale

Only 119 instances / 106 unique phrases carry a reference label, across 7
predicates, with 3 of those having n <= 7. Per-predicate numbers are indicative,
not precise. The dataset is what the frozen run produced; it was not enlarged
with synthetic paraphrases, which would have measured a generator rather than
the benchmark.

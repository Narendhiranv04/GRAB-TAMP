# Alias-resolution replacement study

Auxiliary, post-hoc, offline. **Not part of the FM-TAMP method and not wired
into it.** Nothing here changes runtime behaviour, and no file outside this
directory was modified.

Headline result and full analysis: [`results/experiment_summary.md`](results/experiment_summary.md).

## Question

Can an off-the-shelf zero-shot text classifier map a free-form FM relation
phrase onto the fixed executable predicate interface, without hand-enumerating
lexical aliases?

Answer: **not from the phrase alone** -- 0.376 against 0.419 chance once
forced-choice rows are removed. The frozen mapping turns out to be
endpoint-conditioned rather than purely lexical, which a phrase-only classifier
cannot recover. Details and the one domain where it does work are in the summary.

## Isolation

- No FM or VLM calls; reads only the frozen 320-run artifacts.
- No pipeline file modified. `relation_interpreter.py`, `predicate_registry.py`,
  `fm_schema_v3.py`, `grounding.py`, `robot_capability_registry.py` and
  `structural_sanitizer.py` are byte-identical before and after (hashes in
  `results/isolation_audit.json`).
- The manual baseline is imported read-only rather than reimplemented, so it
  cannot drift from the code that actually ran.
- The frozen 320 outputs are read, never written.

## Data source

`benchmark_reports/full320_s01`, `_s02`, `_s03`, restricted to the trials scored
in `benchmark_reports/FINAL_10x32_RESULTS/final_10x32.json`. 309 of 320 trials
have an archived FM response; the 11 absent are infrastructure outages that
never reached the model, so they are not observations of the method.

## Pipeline

```
extract_frozen_relations.py   frozen trials -> data/frozen_relation_instances.jsonl
                              (also records the frozen manual resolver's disposition)
build_candidate_labels.py     predicate_registry -> data/canonical_predicate_inventory.json
run_manual_baseline.py        baseline predictions + coverage summary
run_zero_shot_nli.py          DeBERTa-v3 zero-shot -> results/zero_shot_predictions.jsonl
evaluate.py                   metrics, confusions, unresolved analysis
```

Run in that order from the repository root with `PYTHONPATH=.`.

## Design decisions worth knowing

**The manual cue table is never shown to the classifier** -- not as label text,
prompt, or example. Candidate hypotheses are built mechanically from the
registry's own name and description. Showing the classifier the aliases whose
necessity it is testing would make the study circular.

**Descriptions were not edited after seeing scores.** `ACCESSIBLE_FROM_BOTH_SEATS`
attracts predictions because its registry sentence is the most generic in its
domain; rewriting it to fix that would be hand-engineering the label text, which
is the practice under test.

**Two evaluation sets are kept apart.** The manual resolver defines the reference
labels on `REFERENCE_RESOLVED`, so its accuracy there is 1.0 by construction and
is reported as coverage instead. `FROZEN_UNRESOLVED` has no labels, so it gets
confidence bands and no accuracy.

**Chance baselines are reported everywhere.** Candidate sets are small -- two
predicates in Kitchen -- so raw accuracy is not interpretable on its own, and
the type-filtered condition raises chance faster than it raises accuracy.

## Environment

Recorded at run time in `results/zero_shot_environment.json`. Model
`MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` revision `b2730f16`,
transformers 4.57.3, torch 2.9.1, CPU. Experiment-local dependencies are in
`requirements.txt`; the production environment was not modified.

# Resume here

Written 2026-09-10, replacing the 2026-09-09 version, which described a grid
that has since been superseded. Anything scheduling this work lives inside a
Claude session and dies with it; the runs do not. They are `systemd --user`
units in `lh.slice` with `Linger=yes` and finish on their own through logout.

## The short version

An audit of all 51,696 JSON artifacts found that **three of the four Kitchen
zeros in the previous table were harness faults, not method results.** They
are fixed at `f352eebe` and the affected cells are being re-run. Do not report
anything from the pre-`fixed_20260910` Kitchen roots.

Read `BASELINE_FIDELITY.md` 2026-09-10 sections before making any claim. It
carries the fault table, what each fault invalidates, and the three items left
unfixed with the reason.

## What is running

Ten legs into `runs/{kitchen,living_room,workshop}/execution/fixed_20260910`,
units `rr-<scene>-<method>`:

    kitchen      vlm_tamp(5w) robust_tamp(4w) owl_tamp(3w) vilain_tamp(3w)
    workshop     robust_tamp(2w) owl_tamp(1w) vilain_tamp(1w)
    living_room  robust_tamp(1w) owl_tamp(1w) vilain_tamp(1w)

**ViLaIn must run under `.venv-vilain-tamp`, not `.venv`.** Under the wrong
interpreter it fails with `live model calls require the dedicated
.venv-vilain-tamp environment` after building the scene, which looks like a
scene error rather than an environment one.

Not re-run, and correctly so: **VLM-TAMP Living Room and VLM-TAMP Workshop**.
An import-graph check confirms neither reaches any file changed by the fixes,
so `runs/living_room/execution/grid_20260907b` and
`runs/workshop/execution/vlm_rerun_20260909` stay canonical for those cells.
Retrieval stays excluded: its Kitchen goal contract is the sentinel
`__planning_goal_unset__` in every variant, and it has one seed per variant
and no Living Room or Workshop runs.

## Verifying a leg

Count canonical seed paths. Do not trust exit codes -- both the batch runner
and ViLaIn have exited 0 with a grid incomplete, twice.

    find runs/kitchen/execution/fixed_20260910 -name benchmark_execution_result.json | wc -l

Then run the three artifact checkers, which assert what the harness itself
claims rather than what a method ought to do:

    .venv/bin/python scripts/audit/check_episodes.py
    .venv/bin/python scripts/audit/check_protocol.py
    .venv/bin/python scripts/audit/check_consistency.py

`check_protocol.py` is the one that proves the decoding condition was uniform
and that nothing private leaked into a prompt.

## Regenerating the tables

    .venv/bin/python scripts/paper_metrics_table.py --validate          # pooled
    .venv/bin/python scripts/paper_metrics_table.py --split --latex     # reported
    .venv/bin/python scripts/planning_metrics.py                        # plan vs execution

`DEFAULT_ROOTS` in `scripts/paper_metrics_table.py` still points at the
pre-fix roots. **Update it to the `fixed_20260910` roots once the legs land**,
keeping the two VLM cells on their existing roots. Partial cells render as
incomplete rather than as a result, so a half-finished leg cannot masquerade
as a zero.

`scripts/planning_metrics.py` is new: it scores the goal conditions against
the plan instead of the terminal state, so a decomposition failure and an
execution failure stop looking alike. Under the pre-fix code Kitchen's gap
between the two was about 51 points.

## Cost, so the next reader budgets correctly

Kitchen wall clock is dominated by physics, not inference. One PLACE that
times out costs **796 s**, measured in `runs/placefix_smoke`. The fix makes
that failure recoverable rather than fatal, which means an episode can now pay
it more than once.

## Still blocked

The branch `baseline_executions` is not pushed. `origin` is HTTPS with no
stored credential on this host, the SSH key is not authorised for it, and `gh`
is not installed. This needs a credential from the researcher; do not work
around it.

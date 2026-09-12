# Resume here

Written 2026-09-12, replacing the 2026-09-10 version, which described ten legs
that have since finished and a set of run roots that no longer exist.

Nothing is running. Scheduling lives inside a Claude session and dies with it;
runs are `systemd --user` units in `lh.slice` with `Linger=yes` and survive
logout, but all of them have been stopped.

## The short version

The canonical grid is **1,280 episodes** in four roots, complete and gap-free:

    runs/kitchen/execution/fixed_20260910        480
    runs/living_room/execution/newgoal_20260911  400
    runs/workshop/execution/fixed_20260910       300
    runs/workshop/execution/vlm_rerun_20260909   100

Those are what `scripts/paper_metrics_table.py:DEFAULT_ROOTS` reads. Every
other root was superseded and was deleted on 2026-09-12.

**Read `BASELINE_FIDELITY.md` before making any claim.** Its 2026-09-12
sections carry the fault table, the budget analysis, and what may be reported.

## What is NOT reportable, and why

`scripts/termination.py` classifies why an episode ended. Budget stops,
inference-server failures and prompt-leakage refusals are not method failures
and must not be published as task-success zeros. Measured on the canonical
grid, four cells contain no method failures at all:

    kitchen  ROBUST-TAMP   0/120 reportable   66 budget, 35 guard, 19 unreachable
    kitchen  VLM-TAMP      0/120 reportable   120 budget
    workshop ROBUST-TAMP   0/100 reportable   47 budget, 53 unreachable
    workshop VLM-TAMP     10/100 reportable   90 budget

76 episodes across the grid died because the inference server was unreachable.
Those need re-running; classification labels them but cannot recover them.

## Goal coverage definition

Fixed conjunction, fixed denominator, defined once in `scripts/goal_coverage.py`:

    Kitchen (12)      coffee x2: coffee_poured, water_poured, stirred, served
                      soup   x2: served, dedicated_utensil_served
    Living Room (5)   personal setting x2: cup_placed, saucer_placed
                      shared: remote_placed
    Workshop (3)      fastener_inserted, joint_repaired,
                      driver_returned_to_workbench

A condition already true at t=0 counts as satisfied. Do not re-derive the
denominator from the ground-truth action list; that is what made L2/L5 score
out of 4 and ViLaIn out of 5 on the same variant.

## Open work, in order

1. **Re-run the 76 infrastructure deaths.** Pure loss, no data.
2. **Kitchen at `--max-model-calls 15`.** At 5 the reported mean is 5.00,
   exactly the cap in 100% of episodes, which measures the protocol. A partial
   re-run sits in `runs/kitchen/execution/budgetfix_20260912` (VLM 15, OWL 46,
   ROBUST 0 of 120 each) and is **not** usable as-is: it is incomplete and
   spans the replan-prompt leakage fix.
3. **Workshop's cap is unresolved.** It exhausts 90% at cap 5 and was still
   hitting the ceiling at 15. `runs/workshop/execution/budget_probe30_20260912`
   was probing cap 30 when everything was stopped.
4. **ViLaIn needs re-running.** `identity.py` no longer aborts an episode when
   one estimate matches no scene entity; it drops it and plans with the rest.
   179 of 320 episodes had usable detections alongside unusable ones and died
   on the unusable ones, so its uniform zero is expected to move.

## Verifying a leg

Count canonical seed paths. Do not trust exit codes -- the batch runner and
ViLaIn have each exited 0 with a grid incomplete.

    find runs/kitchen/execution/<root> -path '*/seed_[0-9][0-9][0-9]/benchmark_execution_result.json' | wc -l

Then the three artifact checkers, which assert what the harness claims rather
than what a method ought to do:

    .venv/bin/python scripts/audit/check_episodes.py
    .venv/bin/python scripts/audit/check_protocol.py
    .venv/bin/python scripts/audit/check_consistency.py

`check_protocol.py` proves the decoding condition was uniform and that nothing
private leaked into a prompt. Decoding is verified uniform across 2,796
requests: one configuration, no drift.

`scripts/audit/repair_root_manifests.py` rebuilds a root's protocol manifest
from its per-episode manifests. Root-level manifests were overwritten by
concurrent legs until 2026-09-12; each leg now writes its own.

## Regenerating the tables

    .venv/bin/python scripts/paper_metrics_table.py --split --latex
    .venv/bin/python scripts/planning_metrics.py
    .venv/bin/python scripts/main_table_metrics.py

## Operational rules that cost time when ignored

- **Never `pkill` or pattern-match kill.** It has matched the calling shell and
  killed it. Stop by exact unit: `systemctl --user stop <unit>`.
- **ViLaIn must run under `.venv-vilain-tamp`**, not `.venv`. Under the wrong
  interpreter it fails after building the scene with a message that looks like
  a scene error.
- **Rebalancing a leg discards its in-flight episodes.** Rebalance once, early,
  from measured wall clock.
- A commit landing mid-run no longer voids ViLaIn episodes; that veto was
  replaced by recorded provenance on 2026-09-12.

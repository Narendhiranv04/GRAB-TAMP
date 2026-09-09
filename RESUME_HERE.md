# Resume here

Written 2026-09-09 so this survives the session that produced it: the schedules
that were driving this work (a 30-minute status cron and a memory/veto
watchdog) live only inside a Claude session and are gone with it.  The runs
themselves are not: they are `systemd --user` units in `lh.slice` with
`Linger=yes`, so they continue through logout and finish on their own.

## Where the grid stands

Complete and verified at 120 or 100 canonical results, with no
`result_present: false` rows in `batch_summary.json`:

  Kitchen      VLM-TAMP, OWL-TAMP, ROBUST-TAMP, ViLaIn-TAMP   (120 each)
  Living room  VLM-TAMP, OWL-TAMP, ROBUST-TAMP, ViLaIn-TAMP   (100 each)
  Workshop     VLM-TAMP, OWL-TAMP, ViLaIn-TAMP                (100 each)

Still running: **Workshop ROBUST-TAMP**, `runs/workshop/execution/robust_rerun_20260909`,
unit `lh-rr-ws-rt`, 10 workers, script in the session scratchpad (copy in
`scripts/` if that is gone).  Verify it by counting canonical seed paths, not by
its exit code -- both this runner and ViLaIn have exited 0 with a grid
incomplete.

Regenerate the table with `python -m scripts.paper_metrics_table --validate`;
partial cells render as incomplete rather than as a result.

## Do not trust an exit code

Two separate legs reported success while silently dropping work.  ViLaIn
Kitchen exited 0 at 65/120 because 55 episodes failed the interpreter's stage
lookup, wrote no result, and were recorded in `batch_summary.json` with
`result_present: false`.  Always count `root/method/variant/images_3/seed_NNN/
benchmark_execution_result.json` against the expected grid.

A commit or a dirty tree vetoes in-flight ViLaIn episodes, so nothing here may
be applied while `lh-vilain-kitchen` is running.

## 1. ViLaIn provenance guard (approved)

Keep every SHA-256 check.  They are the real guarantee: the config, the PDDL
domain, the knowledge file and the run's own artifacts are verified by content,
which is exactly what determines whether "this plan came from that domain" is
true.  `verify_artifact_manifest` already runs unconditionally.

Drop the two repo-wide git assertions that have twice destroyed good data and
cannot affect a PDDL plan:
  - `required_branch` equality -- a branch *rename* would have cost the whole
    Kitchen leg at the same commit with identical file contents.
  - the untracked-path set -- an untracked `.tex` file cost 6 episodes.

Keep HEAD-unchanged and tracked-changes-empty, but evaluate them at run START
as a precondition rather than at the execution boundary, so the check fails in
seconds and fails uniformly for every episode.  Today it is gated on
`planning.status is SUCCESS`, so it deletes only successful episodes and leaves
failures standing as apparent data -- biased in the worst possible direction.

Files: `mujoco_scenes/baselines/vilain_tamp/artifacts.py`
       `mujoco_scenes/baselines/vilain_tamp/runner.py`
       `mujoco_scenes/baselines/vilain_tamp/configs/qwen_only.yaml`
       (`execution_branch` becomes unused -- remove it rather than leave a
        field that looks enforced but is not)
Tests: `mujoco_scenes/baselines/vilain_tamp/tests/` (367 pass today); add a
       case proving a branch rename and an untracked file no longer veto, and
       that a changed domain-file hash still does.
Docs:  section 0 trap entry in `CLAUDE_HANDOFF.md` must be rewritten -- the
       pre-flight check it now prescribes becomes unnecessary.

## 2. Remaining stale docs (from the codebase sweep)

- `MACHINE_HANDOFF.md:60` -- `git checkout baseline_execution`, a branch that
  does not exist; it is `baseline_executions`.
- `MACHINE_HANDOFF.md:98` -- "Expect **5 failures, all Kitchen**".  Stale even
  for the narrower subset command it documents.  Re-measure the subset and
  write the real number; the full suite is 57 failed / 12 errors.
- `MACHINE_HANDOFF.md:67` -- "the suite shows ten failures instead of the
  expected seven" in the mink comment; the counts no longer mean anything.
- `docs/BENCHMARK_PROTOCOLS.md:144` -- "Living Room and Workshop baseline
  runtimes remain planning-only".  Both execute physically now.
- `MACHINE_HANDOFF.md`, `CODE_AUDIT.md` -- present
  `baseline_common.make_paper_tables` as the table generator.  It is Living
  Room only and does not know the Kitchen or Workshop coverage derivations;
  `scripts/paper_metrics_table.py` supersedes it.

## 3. Then

- Regenerate `results_table.tex` from the completed legs and re-run
  `--validate` on all three scenes.
- Re-measure the subset suite count with the machine idle (it was killed
  mid-run for competing with the legs for RAM).

## 4. Workshop ROBUST-TAMP is parked at 24/100 (2026-09-09)

Its re-run was producing worse data than the run it was meant to replace: 1.25
planner transport faults and 1.45 request timeouts per episode against the
retired run's 0.54, with 0 of 24 episodes succeeding.  At 4 workers earlier the
rate was 0.37/ep, so its own long generations were queueing behind each other
under 23-way concurrency; Kitchen RT, whose prompts are smaller, stayed at
0.20/ep throughout.

Resume it ALONE once the other three legs drain, at `--workers 6`, so its
24576-token generations are not competing for decode slots:
  systemd-run --user --slice=lh.slice --unit=lh-rr-ws-rt --property=Type=simple \
    --collect bash -c "$S/rr_workshop_rt.sh >> $S/rr_workshop_rt.log 2>&1"

Sequencing bonus: ViLaIn finishes before this leg starts, and Workshop RT has
no provenance guard, so the commits in sections 1-3 can land BEFORE it runs.
That also makes it safe to raise `timeout_seconds` in
`mujoco_scenes/tamp/discovery_planner.py` (currently 1200.0) if the fault rate
is still high when it runs alone -- a client timeout is infrastructure, not a
decoding condition, and the codebase already treats a transport fault as a
harness fault rather than a planning failure.  Decide from the measured rate,
and record whatever is chosen in BASELINE_FIDELITY.md.

## 5. Measured: ROBUST-TAMP wastes ~90% of its decode (2026-09-09)

For BASELINE_FIDELITY.md, beside the 24576-ceiling limitation.  Measured from
`latency_s` in `model_calls/*.json` across the two ROBUST-TAMP re-run legs,
48.7 GPU-hours of recorded decode:

  cause                                calls   hours   % of all decode
  transport: timeout/unreachable          88    14.7            30.1%
  schema: wrong top-level keys           182    11.5            23.5%
  plan validation: gripper occupied      195    11.0            22.5%
  plan validation: not visible            44     3.7             7.5%
  plan validation: not holding            51     2.7             5.5%
  truncated at the 24576 ceiling           1     0.1             0.3%
  total wasted                           565    43.7            89.8%

Read it in two parts, because they have different owners:

  ~59% is the method's own invalid output -- schema non-compliance plus plans
  rejected by precondition validation.  This is intrinsic to ROBUST-TAMP under
  this decoding condition and is a reportable result.  The schema bucket is
  near-misses: of 54 rejections inspected, 19 omitted `status`, 13 used `plan`
  for `actions`, 5 sent `plan, status`, 5 echoed `output_schema`, 2 sent a bare
  unwrapped action, 1 used `PLAN`.

  ~30% is transport timeouts, which scale with concurrency and are therefore
  partly self-inflicted: Workshop RT reached 1.45 timeouts/episode at 8 workers
  against 0.37 at 4.  Do not report this share as a property of the method.
  Re-measure it from the quiet-GPU Workshop RT leg (section 4).

Truncation at the ceiling is now 0.3% of decode, against 0.53/episode in the
retired run -- so the token ceiling, the thing originally suspected, is not
where ROBUST-TAMP loses its compute.  Schema compliance is.

### Instrumentation gap this exposed

VLM-TAMP and OWL-TAMP record no usable `latency_s` (0 of 395 calls in the
Workshop VLM leg), and no method records token usage at all, so the efficiency
comparison cannot be made across methods -- only within ROBUST-TAMP.  An
earlier table of mine showed VLM-TAMP at "0.0% wasted"; that was missing data,
not a finding, and must not be reported.  Add `latency_s` and the response's
token usage to every method's model-call artifact so "wasted decode" can sit
beside "Raw VLM requests" as a real efficiency column.

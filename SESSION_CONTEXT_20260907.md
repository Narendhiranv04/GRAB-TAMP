# Session context, 2026-09-06/07

Working notes for the baseline-execution and fairness work done on this
machine. Read `BASELINE_FIDELITY.md` for what may be *claimed*; this file
records what was found, decided and left open, so the reasoning survives a
context reset.

## The two fairness defects (most important thing in this file)

**Results produced before commit `44e48f0b` are not comparable across
methods.** Two independent biases, pointing in opposite directions.

1. **The task instruction differed per method, favouring the baselines.**
   The Workshop instruction handed to the baselines named the object
   categories and prescribed the insertion geometry -- "the compatible
   *screw*", "the first compatible *driver*", "*tip-down*", "*head/recess on
   top*" -- where the published Table I instruction says only "the compatible
   *components* required to complete the fastening" and "reusable
   *equipment*". Kitchen's added "Search the closed kitchen storage for
   anything still required", a procedural hint Table I omits.
   Fixed: one source of truth, the scene config each runtime already reads
   (`kitchen_feasibility_variants.yaml:goal_instruction`,
   `living_room_variants.yaml:task`,
   `workshop_variants.yaml:canonical_task_instruction`), re-exported by
   `mujoco_scenes/benchmark_task_instructions.py`. Five stale copies removed.

   Measured effect on Workshop VLM-TAMP: **7/9 success before, ~11% after.**
   The 11% is the honest number.

2. **The proposed method's own prompt leaked the benchmark, favouring it.**
   `SYSTEM_PROMPT` in `mujoco_scenes/workshop_phase1/fm_adapter.py` gave
   role-function examples "contain hot liquid", "stir drink", "drive
   fastener", "hold items for viewer"; the negative example "coffee
   preparation"; and property examples "open cavity", "elongated shape",
   "planar horizontal support" -- three of which *are* published functional
   relations. Fixed: examples now come from outside the benchmark ("cut a
   sheet of material", "illuminate a dark area", "clamp two parts together"),
   and the verifier-capability list keeps every capability while naming no
   benchmark entity. Four leakage guards that had failed since the merge now
   pass.

   **The proposed method's own results need re-running because of this.**

## Exposure: what each method sees

- **Three images to every model.** The grid runs `--camera-counts 3`; ViLaIn
  was given a matching 3-view subset (it had been fixed at 5).
- **The proposed method's detector uses five views** while its language model
  still sees three. That is a property of the method, not an image-exposure
  advantage. Do **not** report it as "five cameras".

## Decisions made, with the evidence

- **`--max-model-calls 5`.** Paired 5-vs-10 trial on Workshop: 4 episodes
  consumed calls 6-10 and **all 4 failed**, at 2178/2220/2369/3277 s against a
  ~890 s median at 5. The one cell that flipped to success used 3 calls, i.e.
  inside a 5-call budget, so it was decoding variance. Consistent with the
  completed Living Room grid: only 6 of 121 episodes ever exceeded 5 calls,
  and 5 of those 6 failed anyway. 10 calls would roughly double the grid
  (~44 h vs ~22 h) for no measured gain. Evidence kept in
  `runs/workshop/execution/calls10_trial`.
- **Workshop runs the assisted-grasp path, not contact-gated.** Contact
  gating was enabled for the full 10-variant suite and measured failing; the
  grasp geometry was calibrated against the assisted path, where the
  attachment constraint absorbs the positioning error. Kitchen and Living
  Room *are* contact-gated. State the difference; do not average it away.
- **Episodes are not reproducible per cell.** `model-native` decoding is
  stochastic (temp 0.6, top_p 0.95), so the same variant and seed can take a
  different trajectory. Pooling 10 seeds per variant captures this; a
  per-cell claim will not replicate.

## Baseline status

| Method | Kitchen | Living Room | Workshop | Notes |
|---|---|---|---|---|
| VLM-TAMP | yes | yes | yes | no receding-horizon protocol exists |
| OWL-TAMP | yes | yes | yes | single-shot, but ~11 raw requests/round |
| Retrieval | new | yes | new | CPU-only (CLIP); no model requests |
| ViLaIn-TAMP | yes | yes | yes | ported this session; see below |
| ROBUST-TAMP | -- | -- | -- | **no implementation exists** |

`model_calls` counts planning *rounds*; `raw_vlm_requests` counts HTTP
requests. OWL-TAMP is 1 round but ~11 requests (sketch + one constraint call
per sketched action), so it is the more expensive method per round despite
never replanning. Its `replans = 0` is structural, not an achievement.

## ViLaIn-TAMP: four blockers cleared, one finding

Ported as the package (76 files) from `naren/ViLaIn-TAMP`, **not** the branch,
which carries 15.6M lines of run artifacts. Its own 357 tests pass unmodified.

Cleared: Naren's absolute Fast Downward/VAL paths (both now built under
`third_party/`, resolution anchored at the repo root); the enforced execution
branch, which was a module constant naming one contributor's branch and is now
`BaselineConfig.execution_branch`; the missing shared artifact; and a
context-window overflow from requesting 4096-8192 output tokens *on top of* a
28-31k prompt.

**Finding (genuine, reportable):** ViLaIn writes `:init` and `:goal` in
separate model calls and they are mutually inconsistent. Living Room L1
generated goals put both cups on one table, the remote on a *personal* table
in one run, and `saucer_1` on two tables at once -- physically impossible.
When init and goal coincide, Fast Downward solves in **0 actions** (a
"zero-step plan", which ViLaIn's own `PAPER_METRICS.md` already excludes from
"Physical Plan Found"); when they conflict, the goal is unreachable. Either
way zero actions execute.

**The candidate presentation is fair** -- verified. The initial-state prompt
offers the full Cartesian product (e.g. all 15 `(at object, table)`
combinations for L1, all 5 locations per object in Workshop), so nothing
discloses the true state. The failure is the model's, not the harness's.

Also note ViLaIn's transport hardcodes `temperature 0` / thinking off, which
is paper-faithful for it but *differs* from the `model-native` condition
VLM-TAMP and OWL-TAMP run. `BASELINE_FIDELITY.md` requires one decoding
condition per table -- unresolved.

## Kitchen execution (fixed this session)

Five of six feasible variants pass strict execution; F0 completes 24/24.
Root cause was **K-02**: the verified-release fallback returned success while
doing only the countertop bookkeeping, so payloads that reached the serving
surface were invisible to `serving_placements`, which five consumers read --
including the slot-overlap guard, meaning the allocator could place a vessel
on top of an already-served one. See `CODE_AUDIT.md` K-02.

Other fixes: acceptance uses the measured opening half-extent rather than the
safety-margined *planning* radius; the oversized spoon (0.264 m, cavity chord
0.147 m) is seated against the far wall; retreat tries outward-and-up; IK
restart seeds are available to every pick.

**Do not retry:** tilting *every* serving utensil into a rim-rest. It was
measured -- fixed F0, broke F1 (tip 0.119 m out) and cascaded into F2's bowl
pick. Scoped to utensils that genuinely cannot fit.

`utensil_bowl_minimum_signed_distance_m` exists because the derived tip point
lies **2.6 cm outside** the collision surface for the oversized spoon, so tip
depth cannot answer whether the utensil intersects the vessel. Measured 0.8 mm
contact, i.e. clean.

## Metrics: what exists and what the table needs

`summarize_execution_batch` emits `outcome_correct_percent`,
`feasible_success_percent`, `infeasible_rejection_percent`, and the `mean_*`
counters. **Not implemented:** goal coverage, false completion, physical plan
found. All three are derivable from artifacts already on disk.

Infeasible variants are scored on **rejection**, not success: `success` is
false by construction there, and credit comes from
`predicted_outcome == "INFEASIBLE"`. The verdict is three-way --
`FEASIBLE` / `INFEASIBLE` / `UNRESOLVED` -- and
`infeasibility_proven()` requires **every** storage region to have been
inspected first, so simply exhausting the budget scores UNRESOLVED rather than
earning a rejection. Do not restrict infeasible variants to planning-only:
proving infeasibility requires physically opening every region.

Goal verification is **not** object-region pairs except in the Living Room.
Kitchen is a conjunction of 7 conditions including two pour relations and a
stir relation; Workshop includes the fastening plus a discovery-discipline
check. `hand_empty_at_end` appears in both and is legitimate -- the Workshop
instruction literally requires leaving the equipment on the bench.

## Infrastructure

- vLLM on gvlab2 via tunnel `127.0.0.1:18000`, `qwen35-9b`,
  `--max-num-seqs 8` (was 2, which made `--workers` pointless: it queued
  instead of batching). Verified 8 concurrent, 0 waiting.
- **Inference is 94-96% of episode wall time.** Physics is noise. The lever
  is server capacity, not local CPU.
- `--workers` added to the batch runner; match it to `--max-num-seqs`.
- Grid throughput ~30-40 episodes/hour; 640 episodes ~16-22 h.
- ViLaIn needs `.venv-vilain-tamp` (enforced at runtime, deliberate
  isolation). Retrieval is CPU-only and can run beside the GPU grid.

## Test suite

~73 problems, essentially all inherited from the Naren merge and **already red
on his branch** (74 of the ~76 merge-introduced failures reproduce there).
Twelve errors need `runs/integrated_no_pot_clearance_seed19_20260807`, part of
the 5 GB `runs/` archive never copied to this machine. Net change this
session: **-3 failures, 0 new.**

Correction worth remembering: the "99 lost Living Room episodes" was a
miscount of failed *model calls*. All 33 interrupted episodes were re-run on
`--resume`; the grid is complete at 180 with zero missing artifacts.

## Line-by-line verification pass (2026-09-07)

Went through the four baseline codebases against `BASELINE_FIDELITY.md` claim
by claim. Confirmed as documented: `PAPER_MAX_SKELETONS = 12` (VLM-TAMP),
`PAPER_MAX_SAMPLES_PER_ACTION = 500` (OWL-TAMP), `--max-sketch-actions 24` at
both call sites, CLIP ViT-B/32 weights, PDDLStream pinned and activating at
`b38137e4`, the three-attempt object reducer
(`goal_related` / `visible` / `all`), and both model-driven baselines sending
byte-identical `model-native` sampling. Six defects found:

1. **Workshop GT normalization discarded every expected action.**
   `normalize_workshop_actions` mapped the long-form vocabulary
   (`INSPECT_STORAGE`, `DRIVE_FASTENER`, ...) but the frozen
   `EXPECTED_GT/workshop` files use `OPEN`, `PICK`, `PLACE`, `SCREW`. The
   retention pass keys off *normalized* names, so nothing was retained: the
   normalized expected sequence was empty in **all 95** recorded artifacts,
   and where the baseline also produced nothing that scored
   `exact_sequence_match: True` with `ordered_f1: 1.0` -- **a perfect score
   for doing nothing**, in 11 of 114 artifacts.
   Fixed via `_WORKSHOP_OPERATOR_SYNONYMS` plus a region-keyed
   `PLACE(item, non_region_target) -> INSERT` rule. All ten variants now
   normalize to complete skeletons; W9/W10 correctly reduce to the
   three-region exhaustive inspection.
   **`success` and `outcome_correct_percent` are unaffected** -- they come
   from the goal verifier, not from this comparison -- so no episode needs
   re-running. The 114 artifacts are recomputable from the raw sequences they
   already store: `scratchpad/repair_gt_comparison.py --apply` (keeps the
   original as `gt_sequence_comparison.pre_normalizer_fix.json`). **Not yet
   applied.**
2. **Two tests were stale in the same way** and had been red, masking the
   above: they checked for raw `INSPECT_STORAGE` / `DRIVE_FASTENER` in GT
   files that contain neither. Now read through the normalizer, plus two new
   guards -- every frozen GT trace must normalize non-empty, and an empty
   prediction must never score a perfect match.
3. **ViLaIn-TAMP hardcoded greedy decoding** (`temperature 0`, thinking off)
   at the transport, so it could not run the table's condition. Now
   selectable: `RunOptions.decoding` / `--decoding`, with `paper` as the
   default and `model-native` drawing from `baseline_common.inference`, the
   same source the other three read (asserted equal). The grid passes the
   table's condition through. This is the likely cause of ViLaIn's 0%: Qwen's
   own card advises against greedy decoding for this checkpoint, and
   near-greedy sampling is where plan degeneration was first observed.
   Its 357 tests still pass.
4. **The grid dropped `--camera-count` for ViLaIn.** Harmless today because
   its runner defaults to 3, but a `--camera-counts 1,3` grid would have run
   both cells at 3 and labelled the artifacts as though they differed.
5. **The Workshop observation contract hardcoded
   `"physical_execution": False`** while the grid passes `--execute`. Now
   derived from the actual mode at all three call sites. The authoritative
   artifacts (`benchmark_execution_result.json`, `method_manifest.json`)
   always said `true`, so no reported number was wrong -- but episodes
   recorded before this fix carry the stale contract field.
6. **Stale fidelity claims corrected:** "Workshop is planning-only" (it is
   executed), "a five-image VLM query" (OWL sends `--camera-count`; the grid
   sent three), "at most five skeletons" (exactly one is ever explored),
   Kitchen "planning-only trials", and the annotation-policy table, which
   omitted Workshop.

**Workshop annotation is worth restating for the paper.** `OBJECT_LABELS`
gives the model "screw", "manual screwdriver", "power screwdriver", "wooden
hammer"; `REGION_LABELS` gives "left drawer", "tool cabinet", "main
workbench". So the category nouns the de-biased instruction removed come
straight back through the observation layer. Faithful to both papers, and
equal across them -- but a real advantage over the proposed method and the
retrieval control, and it means only the *compatibility* relation is left for
the model to infer.

**Kitchen retrieval is verified working but very slow.** The smoke test opened
C1, D1 and D2 correctly (`Found: []`, right for K1) and then hit a 3000 s
timeout before scoring: roughly 16 min per physical drawer open, because
Kitchen's `INSPECT` routes through `phase_a.request("OPEN", ..., execute=True)`
-- a full base-motion + IK drawer open, five of them before any CLIP scoring.
At that rate a 12-variant x 10-seed Kitchen retrieval grid is ~240 h and is
**not** feasible for the deadline. Workshop retrieval is fine
(`RETRIEVED, 5 actions`, correct `FEASIBLE` verdict, ~minutes). Timing probe:
`scratchpad/time_inspect.py`.

## Metrics recording bug (found 2026-09-07, fixed, backfilled)

**Only the Workshop runners passed `expected_outcome`/`predicted_outcome` to
`write_execution_result`.** Living Room and Kitchen, for *both* VLM-TAMP and
OWL-TAMP, computed the verdict and wrote it to `gt_sequence_comparison.json`
but never into the shared artifact the metrics read. Retrieval Living Room had
the same gap. Effect: `outcome_correct_percent` and
`infeasible_rejection_percent` were unavailable for two of three scenes, and
every infeasible variant scored as a plain failure.

Fixed in all four runners (AST-verified into the correct call) and guarded by
`baseline_common/tests/test_every_runner_records_the_verdict.py`, which asserts
it structurally across all nine runners -- a missing keyword argument cannot
fail loudly at runtime.

**Backfilled, not re-run:** 170 artifacts recovered the verdict from their own
`gt_sequence_comparison.json` via `scratchpad/backfill_verdicts.py --apply`
(originals kept as `.pre_verdict_backfill.json`). This is why Living Room went
from an apparent `outcome_correct 0/46` to **44/50 (88%)** -- the 0 was the
bug, not the method.

## Results so far (2026-09-07, mid-grid)

| scene | method | n | outcome correct | feasible success | infeasible rejection | median |
|---|---|---|---|---|---|---|
| Workshop | OWL-TAMP | 100 | 13% | 0/80 | **13/20** | 215 s |
| Workshop | VLM-TAMP | 95 | 13% | 8/75 | 4/20 | 889 s |
| Living Room | VLM-TAMP | 50 | **88%** | 44/49 | -- | 266 s |

Two things worth carrying forward. **Workshop is far harder than Living Room
for the same method** (VLM-TAMP 13% vs 88%), which is the discovery
requirement doing the work. And **OWL-TAMP rejects infeasible variants better
than VLM-TAMP** (13/20 vs 4/20) while never solving a feasible one -- its
single-shot sketch tends to be "inspect everything", which is exactly what
`infeasibility_proven()` needs. Do not summarize OWL as uniformly worse.

## OWL-TAMP receding horizon (added 2026-09-07)

The loop returned immediately on `NO_PLAN`, so `--max-replans` was unreachable
and the protocol collapsed onto single-shot: one malformed sketch ended the
episode at round two. Added `replan_on_no_plan` (default **off**, preserving
the strict reading; the row runs with it **on**), which lets a plan-less round
consume a replan and re-plan. `no_plan_rounds` is recorded so a capability
failure is always distinguishable from a budget overrun.

Measured on W1 with `--max-replans 14`: **15 rounds, 14 of them producing no
satisfiable sketch**, 21 raw requests, 1698 s, 1 executed action,
`REPLAN_BUDGET_EXHAUSTED`. So OWL still fails Workshop -- but now
demonstrably because its sketches are unsatisfiable under the `Executed(i)`
constraint, not because it was denied a second chance. The observed failure is
a duplicate-argument sketch such as
`FASTEN(object_0005, object_0005, object_0005)`, which no symbolic plan can
contain. Note the budget (15 rounds) **exceeds** the expected action count (6),
so this is *not* a budget-limitation artifact and may be reported as a genuine
failure.

Cost: ~28 min/episode against single-shot's 215 s. A full three-scene row is
~320 episodes.

## Open questions

1. ~~ViLaIn decoding~~ **resolved 2026-09-07.** `--decoding` selects the
   condition; the grid passes the table's. ViLaIn still needs re-running under
   `model-native` before its column can go in the same table as the others,
   and its `paper` (greedy) numbers become the ablation.
2. Goal coverage / false completion / physical plan found need implementing.
3. `VLM-TAMP (receding horizon)` and `ROBUST-TAMP` table rows have no
   implementation.
4. Kitchen F2 currently fails at 22/25 on the shallow-bowl pick, having traded
   places with F0; 5/6 either way.
6. **Kitchen retrieval throughput.** ~16 min per physical drawer open makes a
   full Kitchen retrieval grid infeasible. Options, cheapest first: cache one
   opened-state snapshot per variant and restore it instead of re-opening
   physically (the opens are deterministic and identical across seeds, and
   retrieval's CLIP scoring is what is under test, not its drawer-opening);
   run fewer seeds for this column since it has no decoding stochasticity --
   **CLIP scoring is deterministic, so seeds beyond the first add nothing at
   all for retrieval** and 12 episodes would suffice instead of 120; or drop
   the Kitchen retrieval cell and say so. Recommend the deterministic-seed
   argument: it is honest, needs no code, and cuts 240 h to ~24 h.
7. **Whether to apply `repair_gt_comparison.py`** to the 114 recorded Workshop
   comparison artifacts (see defect 1). Non-destructive, backs up originals.
5. The proposed method needs re-running after the prompt de-bias.

---

# Session 2, 2026-09-07 evening: the OOM, and the Kitchen fix that unblocks 240 episodes

## What killed the previous session

The **OOM killer killed the entire VS Code snap cgroup at 17:48:04**
(`snap.code.code-*.scope: Failed with result 'oom-kill'`, `27.0G memory peak`
on a 31 GB host with 2 GB swap). Everything launched with `nohup` from the
Claude Code shell was a child of that cgroup, so the extension host, the
Claude session and every running episode died together.

The trigger was oversubscription: four legs at once, the Kitchen one at
`--workers 14`, while the server is capped at `--max-num-seqs 8`. Workers past
8 do not batch -- they queue in vLLM while still holding a full MuJoCo scene in
local RAM. Pure memory cost, zero throughput.

**Nothing was lost.** The transcript is intact at
`~/.claude/projects/-home-longhorizon-Documents-LH-Extension/dced8008-8eec-4369-80a7-41d50636d710.jsonl`
(19 MB) -- filed under the *parent* directory because that session's cwd was
`/home/longhorizon/Documents/LH_Extension`, which is why it does not appear in
the extension's session list when the workspace is opened at `V1`. Its
scratchpad survives too, at
`/tmp/claude-1002/-home-longhorizon-Documents-LH-Extension/dced8008-.../scratchpad`,
with every leg script, log, and `repair_gt_comparison.py` /
`backfill_verdicts.py` / `time_inspect.py`.

**Launch grids as `systemd-run --user` units from now on**, one per leg, with
`systemctl --user set-property <unit> MemoryHigh=9G MemoryMax=12G`. They then
live in `app.slice`, not the snap scope: a grid OOM cannot kill Claude, and
Claude crashing cannot kill a grid.

## Why all 240 Kitchen episodes crashed, and the fix

`runs/kitchen/execution/grid_20260907b` recorded 240 episodes with
`return_code 1` and **zero artifacts**. Two causes, both in
`mujoco_scenes/baseline_kitchen_runtime.py`, both fixed in the working tree
(uncommitted at the time of writing):

1. **K1--K6: `PromptLeakageError ... regions=['B1','C1','C2','D1','D2']`.**
   The shared observation contract has always claimed
   `PERSISTENT_REGION_ID_ONLY`, but the "persistent id" being published was the
   oracle's own region name -- and those five names are on the forbidden list
   in `functional_tamp_pipeline/audit.py` precisely because naming them
   discloses the storage layout, which is Kitchen's entire search problem. The
   transport-level guard added earlier this session (`assert_no_prompt_leakage`
   in `baseline_common/inference.py`) is what turned a silent fairness defect
   into a hard failure. **The guard was right; the Kitchen observation was
   wrong.**
   Fixed by `kitchen_public_region_ids()`: `region_0001..region_0005`, sorted so
   the mapping is identical across methods, seeds and processes, with the
   inverse map kept so actions arriving in the public vocabulary still execute
   against the canonical one. `countertop` and `serving_area` keep their names
   -- the task instruction names them, so they disclose nothing.
2. **K7--K12: `ValueError("Live discovery execution currently requires a
   FEASIBLE variant")`.** An infeasible variant has no satisfiable terminal
   state, so `goal_contract_from_expected_actions` refused to build one and
   aborted the episode before its first model call. Now raises the typed
   `InfeasibleVariantHasNoGoalContract` and the caller substitutes the
   unset-goal contract, so the episode runs and can be scored on whether it
   correctly *rejects* the variant -- which is how infeasible variants are
   scored anyway (`success` is false by construction; credit comes from
   `predicted_outcome == "INFEASIBLE"`).

**Verified end to end**, not just by unit test: a K1 + K7 smoke pair reached
planning and physical execution with **zero** `PromptLeakageError`, zero
FEASIBLE-aborts and zero tracebacks, and `latest_observation.json` for K7
contains `region_0001..region_0005` plus `countertop`/`serving_area` and **no
occurrence of any of B1/C1/C2/D1/D2**. The three test files that cover this
code (`mujoco_scenes/tests/test_baseline_kitchen_runtime.py`,
`vlm_tamp_baseline/tests/test_kitchen_planning.py`,
`baseline_common/tests/test_every_runner_records_the_verdict.py`) pass 24/24.

## Grid state at restart (18:07)

| scene | done | of | remaining |
|---|---|---|---|
| Workshop | 195 | 200 | vlm_tamp W1 s2, W1 s8, W3 s7, W6 s3, W8 s8 |
| Living Room | 142 | 200 | owl_tamp L5 s2-9, L6-L10 all seeds (58) |
| Kitchen | 0 | 240 | everything, now unblocked |
| **total** | **337** | **640** | **303** |

Living Room `vlm_tamp` and Workshop `owl_tamp` are both complete at 100/100.

Relaunched as four units, `--resume --continue-on-error`, flags matched to each
scene's recorded `protocol_manifest.json` so artifacts pool (`--workers` is not
part of the manifest, so lowering it changes nothing reportable):
`lh-kitchen` (workers 6), `lh-lrowl` (4), `lh-wsgap` (3), `lh-ksmoke`.
Status: `scratchpad/status.sh`.

## Test-suite baseline on this machine

**57 failed, 1660 passed, 1 skipped, 12 errors.** `CLAUDE.md`'s "expect
exactly five failures" is **stale** -- it predates the Naren merge. The count
here is consistent with this file's earlier note (~73 inherited problems, net
-3 this session, 0 new):

- The **12 errors** are all one missing fixture,
  `runs/integrated_no_pot_clearance_seed19_20260807/object_registry.json`, part
  of the 5 GB `runs/` archive never copied to this machine.
- **None of the 57 failures touch `baseline_kitchen_runtime`.** The only two
  test files that reference it both pass, so the Kitchen fix introduces no
  failures.

## Plan change 18:20: Kitchen deferred to the Blackwell host, ViLaIn queued on the 5090

The current inference host is an **RTX 5090, 32607 MiB**, which is why vLLM
serves `--max-model-len 32768`. Kitchen's ViLaIn prompt
(`fixed_full_inspection`) runs **24-31k on its own**, so a 16384-token output
budget cannot be honoured there and `_completion_with_context_retry` has to
shrink every request. The incoming Blackwell host (RTX 5000 Pro, 48 GB) lifts
that ceiling, so **all of Kitchen waits for it**; `lh-kitchen` was stopped at
18:14 with 0 artifacts written, so nothing was lost and `--resume` will start
it clean.

Note that Kitchen's *vlm_tamp/owl_tamp* 240 episodes are **no longer blocked**
-- the leakage and infeasible-variant bugs are fixed and verified. They are
waiting on the Blackwell host for throughput (48 GB supports a higher
`--max-num-seqs` than the 5090's 8), not for correctness.

**When the Blackwell host arrives, run the episodes from *this* machine against
its endpoint.** MuJoCo physics is local; the server only does inference. Keeping
the episode host constant keeps `host_cpu` constant across scenes, which is what
`make_paper_tables` cares about. Running episodes *on* the new box instead would
put Kitchen on a different physics host than Workshop and Living Room.

Queued now on the 5090, both `--decoding model-native`, 10 variants x 10 seeds,
3 workers each, `runs/{workshop,living_room}/execution/vilain_20260907`:
`lh-vilain-workshop`, `lh-vilain-living_room`. This is the re-run open question
1 asked for. Verified from the first episode's `model_metadata.json`:
`decoding: model-native`, `temperature 0.6 / top_p 0.95`,
`thinking_enabled: true`, `max_tokens 16384`, `finish_reason: stop` (no
truncation on Workshop), `camera_count 3`. Its greedy `paper` numbers become
the ablation.

**Measured ViLaIn cost:** ~107 s per model call with thinking on, packing
12 source images into 4 contact sheets (one per inspection stage). Hence
`--episode-timeout 5400` rather than the 3600 the vlm/owl grids used.

# Resume 20260908 (afternoon) -- Kitchen POUR fixed, ViLaIn leak fixed

## Commits made this session (branch `kitchen-workshop-integration`, not pushed)

- `2a70ff7a` Publish ViLaIn inspection regions as aliases, not canonical names
- `22ee937a` Let Kitchen baselines execute the POUR/STIR pair they grounded

Both verified; tree clean. Worktree `kitchen-pour-fix` at
`scratchpad/wt-pour` was used so the running ViLaIn legs kept a clean tree
(ViLaIn vetoes an episode that reaches execution if the tree is dirty).

## 1. ViLaIn canonical-region leak -- FIXED, VERIFIED

`prompt_observation_payload` published raw `LEFT_DRAWER` / `D1`, which the
shared leakage guard forbids. Workshop died outright; Kitchen leaked the same
way through `stage_id` ("001_d1") and the embedded `rgb_path`, both of which
slipped past the case-sensitive guard. Now publishes the aliases the other
baselines use, with a drift test against the authoritative maps.
Verified in a real request: 0 canonical tokens, each alias exactly once.
**This unblocked the entire ViLaIn Workshop leg (100 episodes).**

## 2. Kitchen POUR/STIR -- FIXED, VERIFIED END-TO-END

Kitchen feasible-success of 0/60 was a **harness artifact**, not a capability
result. Phase C is a frozen-plan executor; `BaselineKitchenRuntime` reused it
with an empty plan and a registry holding only object ids, so
`expected_pairs == {'POUR': {}, 'STIR': {}}` and every pour failed
`POUR_TARGET_RESOLUTION_FAILED` before geometry was consulted. 368 attempted
pours across 252 episodes, 0 successes. The GT runner never exercised the path
either (`dummy_registry, []` + `OraclePhaseCLedger`).

Fix: Phase C keeps the frozen-plan contract as its default; the baseline path
selects a `GEOMETRIC` policy, judged by opening geometry, reachability and
physics, with a verified-motion ledger. Registry openings now measured from
the instantiated mesh rim via `extract_exact_object_geometry` (perceptual, not
plan-derived). The cupboard-exclusion rule survives and became live, tracked
through PLACE, because the mug starts in `C2` on K2/K4/K6.

Verified on K1: `pick kettle -> POUR_MOTION_VERIFIED, effects applied`.
8 new tests in `mujoco_scenes/tests/test_kitchen_phase_c_admissibility.py`.

## 3. Test suite baseline is NOT five failures any more

`CLAUDE.md` says expect exactly five. Actual on this machine:
**57 failed, 1727 passed, 12 errors.** None are mine (verified: my diff touched
3 files, none imported by the failing tests). Breakdown:
- 44 in `test_phase3_6a*` + `functional_tamp_pipeline` -- the proposed method's
  canonicalization layer, which Naren is actively making functional.
  `AmbiguousCanonicalizationError`, `UnmappedFunctionalConceptError`.
- 13 (1 failed + 12 errors) in `test_kitchen_phase_c_execution.py`, all
  `FileNotFoundError` for `runs/phaseC_plan_contract/phaseC_plan_contract.json`
  and `runs/integrated_no_pot_clearance_seed19_20260807/`. **Both directories
  are missing from this machine** -- `runs/` is not in git and was never fully
  copied. These Phase-C GT tests cannot run here at all.

## 4. Legs running (all in `lh.slice`, ~7.5 GB of 20 GB)

| unit | scene | notes |
|---|---|---|
| `lh-rt-workshop` | ROBUST-TAMP Workshop | correct with no `--headless`: renders offscreen. Low CPU is model wait (`x11=0`, state S/do_poll), NOT viewer throttle -- checked. |
| `lh-rt-living-room` | ROBUST-TAMP Living Room | 52 results, headless |
| `lh-vilain-lr-b` | ViLaIn Living Room | fresh root `vilain_20260908b` under clean commit |
| `lh-vilain-ws-b` | ViLaIn Workshop | fresh root `vilain_20260908b`, leak fixed |
| `lh-kpour-smoke` | Kitchen vlm_tamp K1 s0 | verifying POUR in the real pipeline |

Old ViLaIn Living Room root `vilain_20260908` was abandoned (9 episodes, all
EXHAUSTED, 0 reached execution) so the grid is attributable to one commit.

## 5. OPEN -- ViLaIn fails at entity resolution, not planning

Every ViLaIn episode so far is `EXHAUSTED` (15+ episodes, 0 successes). The
terminal failure is **not** a planning failure:

```
"kind": "ENTITY_RESOLUTION", "reason_code": "UNRESOLVED_ENTITY",
"summary": "no class-compatible visible entity is close enough to 'red_cup_1'",
"candidate_entities": [], "object_ids": ["red_cup_1"]
```

`plannable: true`, `symbolic_plan_length: 10`, `identity_binding_count: 0`.
So ViLaIn plans fine and then cannot bind its own symbolic entity
(`red_cup_1`) to any observed object. This smells like a second harness fault
of the same family as the two already fixed -- a naming/annotation mismatch
between what ViLaIn's planner invents and what the observation publishes --
**but it is not yet diagnosed.** Do this next; until then ViLaIn's numbers are
not trustworthy as a method result.

## 6. Not done

- Kitchen grids not re-run yet (240 episodes). Needs worker budget: Kitchen
  episodes are ~1.4 GB each and 12 concurrent episodes are already running.
- gvlab2 deliberately left idle: vLLM queue depth is 0, so inference is not
  the bottleneck; local RAM is. A second endpoint would add nothing.
- Ablation instrumentation (Regions Inspected, Candidate Checks, Grounding
  Time) still has zero implementation; "No verification" missing from
  `COMPONENT_MASKS`.

## 7. Later the same afternoon -- ViLaIn root cause found and fixed

`e2a49e2d` Stop downscaling ViLaIn observation frames before tiling.

The ViLaIn `EXHAUSTED` streak was **not** a planning failure. ViLaIn packs one
contact sheet per stage and shrank each 640x480 frame to 448x336 first, leaving
a cup ~18x13 px. ViLaIn is the only baseline needing a pixel-accurate box: it
unprojects the box centre through depth to bind entities. A box that misses the
object samples the floor behind the table, so Z collapsed.

Measured on L1, distance from each estimate to its own true object
(limit is `maximum_distance_m = 0.75`):

| object | before (448x336) | after (native 640x480) |
|---|---|---|
| red_cup_1 | 0.941 FAIL | **0.255 OK** |
| red_cup_2 | 1.045 FAIL | **0.137 OK** |
| white_saucer_1 | 0.865 FAIL | **0.322 OK** |
| white_saucer_2 | -- | **0.329 OK** |
| black_remote_1 | 0.827 FAIL | 2.072 FAIL |

Z went from ~0.006 to ~0.70-0.74 against a true 0.795.

**Caveat, not yet resolved:** identity resolution raises on the *first*
unresolved estimate, so the remote alone can still end an episode. 4 of 5
entities now bind; ViLaIn is not yet confirmed to produce a success. The
remote is a genuine model localization miss, not a harness fault.

Diagnosis was empirical: project the known candidate centroids into
`l2_camera_top` and compare with the model's boxes. X was accurate (the remote
projected to exactly 320.0 vs the model's 320.0); Y was systematically ~50-60 px
high. Depth estimator choice was ruled out first -- median, p15, p05 and min all
gave Z~0.01, proving the box was on empty floor rather than the box being noisy.

## 8. Kitchen POUR confirmed working in the real pipeline

`lh-kpour-smoke` (K1, vlm_tamp, seed 0) finished rc=0 and the log shows
`POUR_APPROACH` x18, `POUR_ORIENTATION_SELECTION`, `POUR_GRASP_POSE_HOVER`,
`POUR_ALIGNMENT_FAILED` x4 and one **`POURED`** effect committed. So POUR now
performs real physical motion and can commit its symbolic effect, where before
it returned `POUR_TARGET_RESOLUTION_FAILED` instantly, 368 times out of 368.
The episode itself still ended `UNRESOLVED` (goal not satisfied) -- that is now
a genuine physics/capability outcome rather than a harness block.

## 9. ROBUST-TAMP Living Room finished

`rc=0` at 12:30, **60 results** in `runs/living_room/execution/robust_tamp_20260908`.
**Check this:** the launcher requested L1-L10 x seeds 0-9 = 100 episodes but 60
landed, and the runner still exited 0. Reconcile before reporting that column.

## 10. Running at handoff

`lh-rt-workshop`, `lh-vilain-lr-c`, `lh-vilain-ws-c` (both fresh roots
`vilain_20260908c` under commit `e2a49e2d`), `lh-vilain-restest2`.
Roots `vilain_20260908b` are abandoned (pre-fix, worthless).

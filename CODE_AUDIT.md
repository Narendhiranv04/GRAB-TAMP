# Code audit, 2026-09-05

A file-by-file review of this repository for correctness bugs, doc/code
inconsistencies, dead or misleading code, and duplication. Written so that
someone reading it later can see **what was changed, why, and what it affects**
without having to reconstruct the reasoning from a diff.

## How to read this

Every finding has a severity and a status:

| severity | meaning |
|---|---|
| **RESULT-AFFECTING** | Changing this changes numbers that have already been reported. Read these first. |
| **CORRECTNESS** | A real bug: wrong output, silent failure, or a contradiction between modules. |
| **MISLEADING** | The code works but implies behaviour that does not happen; a reader draws a false conclusion. |
| **DOC** | Prose, comment or docstring that no longer matches the code. |
| **QUALITY** | Duplication, naming, structure. No behavioural change. |

| status | meaning |
|---|---|
| **FIXED** | Changed in this pass; the entry says exactly what and why. |
| **LOGGED** | Found and described, deliberately not changed; the entry says why not. |
| **DECISION** | Needs a call from the researcher before anything is safe to do. |

## Provenance: the state these findings start from

- Branch `baseline_execution`, HEAD `5cd8231`.
- The 180-episode Living Room grid in
  `runs/living_room/execution/feasible_i9_20260904{,_retrieval}` was produced
  by the code at `5cd8231` **plus** the working-tree changes listed in the
  "Pre-audit changes" section below, none of which touch execution behaviour.
- Baseline test state at the start of this audit: **5 failed, 1032 passed,
  1 skipped** (was 7 failed before three stale tests were updated; see
  `MACHINE_HANDOFF.md`).

**If a fix below is marked RESULT-AFFECTING, the grid's numbers were produced
by the code as it stood *before* that fix.** To reproduce them exactly, check
out `5cd8231` and re-apply only the pre-audit changes.

## Pre-audit changes (made before this audit began, for context)

These closed open items 1 and 4 from `CLAUDE_HANDOFF.md` section 0 and fixed
the failure-analysis table. None of them alter how an episode executes, so the
grid remains valid under them.

1. `baseline_common/physical_benchmark.py` -- `write_execution_result` now
   records `expected_outcome`, `predicted_outcome` and `outcome_match`.
   Physical `success` cannot express a correct rejection of an infeasible task,
   so without these a method that correctly refuses an impossible variant
   scored identically to one that blundered through it. Optional, so older
   artifacts stay valid.
2. `baseline_common/summarize_execution_batch.py` -- reports
   `feasible_success_percent` and `infeasible_rejection_percent` separately
   instead of one pooled rate, and back-fills the GT verdict from the sibling
   `episode_result.json` for episodes written before (1). Schema bumped to 2.
   This is what makes L7--L10 runnable as scored work.
3. `baseline_common/make_paper_tables.py` -- failure categories now also match
   the upper-case terminal statuses that `physical_terminal_status` emits.
   Runners that do not pass `terminal_failure` (owl_tamp, retrieval) left no
   lower-case code to match on, so **26 of 81 failures fell into the
   uncategorised bucket**; it is now 81 of 81. Truncation is excluded from the
   plan-rate denominator rather than counted as a failure to plan.
4. `owl_tamp_baseline/planner.py` -- a token-ceiling truncation now reports
   `MODEL_OUTPUT_TRUNCATED` rather than `INVALID_MODEL_OUTPUT`, matching the
   treatment VLM-TAMP already had. Closes open item 4.
5. `baseline_common/run_baseline_execution_batch.py` -- `protocol_manifest.json`
   now records `max_sketch_actions`, `max_replans`, `max_actions`,
   `max_tokens`, `episode_timeout_s`, and the variant/camera/seed lists. Every
   knob that bounds a method's search is part of the reported condition.
6. `mujoco_scenes/functional_tamp_pipeline/grounding.py` -- added an opt-in
   `count_valid_assignments`, removed a dead sort, and stopped reporting a
   `valid_assignment_count` that was structurally always 1. See finding
   [G-01](#g-01) for why this matters to the evidence ablation.
7. Three stale tests updated against the contracts the code implements, and
   `BASELINE_FIDELITY.md` gained the self-collision allowance record.
   Docs corrected in `MACHINE_HANDOFF.md`, `CLAUDE.md`, `CLAUDE_HANDOFF.md`.

---

# Findings

## Automated sweep across all 351 Python files

Run first so that whole classes of defect could be ruled out rather than
sampled. Results, for the record:

| check | result |
|---|---|
| syntax errors | **0** |
| bare `except:` | **0** |
| mutable default arguments | **0** |
| `is`/`is not` against a literal | **0** |
| `== None` | **0** |
| duplicate top-level definitions in one module | **0** |
| `TODO`/`FIXME`/`XXX`/`HACK`/`BUG` markers | **0** |

That is an unusually clean result for a research codebase of this size, and it
is worth stating plainly: the defects below are not of the "someone was
careless" kind. They are consistency and provenance problems, which is what
tends to survive in code that has been carefully maintained but has grown.

`except: pass` appears 41 times and `except Exception` 145 times. Both are
mostly legitimate -- renderer teardown, best-effort file reads with a defined
fallback, optional viewer paths. They were reviewed on the physical-execution
path rather than in bulk; see [E-01](#e-01).

---

<a id="c-01"></a>
### C-01 -- Controller tolerances are redefined in three modules, guarded in one place -- **QUALITY / CORRECTNESS-risk** -- FIXED (test added)

**What.** These are declared independently in more than one module rather than
imported from a single source:

| constant | value | modules |
|---|---|---|
| `BASE_COMMAND_TOLERANCE` | 0.002 | `generic_manipulation`, `mobile_motion`, `living_room_dusting` |
| `BASE_LINEAR_COMMAND_SPEED` | 0.25 | `generic_manipulation`, `mobile_motion` |
| `BASE_YAW_COMMAND_SPEED` | 0.60 | `generic_manipulation`, `mobile_motion` |
| `ARM_COMMAND_TOLERANCE` | 0.001 | `generic_manipulation`, `living_room_dusting` |
| `COLLISION_GUARD_INTERVAL` | 5 | `generic_manipulation`, `living_room_dusting` |

**Why it matters.** These decide when a commanded pose counts as reached. A
silent drift between the manipulation and navigation copies would change
physical outcomes without changing any test that names it. The team clearly
knows: `test_robot_profiles.py` already imports the two *speeds* under aliases
purely to assert they are equal. But that guard covered 2 of the 5.

**Why not consolidated.** There is no circular-import reason for the
duplication -- neither module imports the other, and both already import
`robot_profiles`, which would be the natural home. But moving constants inside
the physics modules that every episode runs through, days after a 180-episode
grid, risks more than the hypothetical divergence it prevents.

**Fixed instead by** extending the existing guard to all five constants:
`test_controller_tolerances_agree_across_the_modules_that_redefine_them`. Zero
behavioural change, and a drift now fails immediately and loudly.

**Recommended later**, when no grid is in flight: move all five into
`mujoco_scenes/robot_profiles.py` and import them, which both modules already
do for other symbols. Then delete the guard test.

---

<a id="c-02"></a>
### C-02 -- Constants that look divergent but are correct -- **no action, recorded to stop the next reader "fixing" them**

The same sweep flagged these as one name with different values. All are
intentional and must **not** be unified:

- `PAPER_MAX_SKELETONS` -- 12 in `vlm_tamp_baseline/pddlstream_refiner.py`, 5 in
  `owl_tamp_baseline/refinement.py`. These are the two papers' **own** published
  values (`BASELINE_FIDELITY.md`: VLM-TAMP "at most 12 diverse skeletons per
  attempt"; OWL-TAMP "at most five skeletons"). Making them equal would break
  algorithmic fidelity to the methods being compared.
- `PROMPT_VERSION` -- 9 / 4 / 3 / 1 across the vlm_tamp, llm3, owl_tamp and
  inference_server prompts. Per-prompt versions; they are supposed to differ.
- `SCHEMA_VERSION` -- per-artifact, not global.
- `SYSTEM_PROMPT`, `TASK` -- per-domain and per-method by design.
- `ARM_COMMAND_SPEED` (0.45 vs 1.20), `BASE_SETTLE_SPEED`, `BASE_SETTLE_TICKS` --
  the lower values are `living_room_dusting`, a separate demo task with its own
  calibration, reached only through `living_room_actions.py`. Not on the
  benchmark execution path.

---

<a id="v-01"></a>
### V-01 -- `validate_repository.sh` could not pass, and did not cover the code that produces the results -- **CORRECTNESS** -- FIXED

Two independent problems in the repository's own canonical check.

**It could not pass.** The script runs bare `pytest` under `set -eu`. The
documented known-good state has failing Kitchen tests, so pytest exits non-zero
and the script aborts every time, on a healthy checkout. A validation script
that always fails is one nobody runs, and then it protects nothing.

**It covered 13 of 70 entry points, and none of the ones that matter most.**
Every module that produces a reported number was absent:

```
baseline_common.run_baseline_execution_batch    baseline_common.make_paper_tables
baseline_common.summarize_execution_batch       baseline_common.run_plan_gt_batch
baseline_common.run_discovery_execution_batch   vlm_tamp_baseline.run_living_room
owl_tamp_baseline.run_living_room               retrieval_baseline.run_living_room
mujoco_scenes.run_gt_evidence_ablation          mujoco_scenes.run_living_room_discovery_replanning
```

A break in any of these invalidates a grid, and the check that was supposed to
catch it never looked.

**Fixed.** The module list now covers 26 entry points including all of the
above, and the five known Kitchen failures are `--deselect`ed by node id rather
than tolerated wholesale -- so the script passes on a healthy tree and still
fails on any *other* failure, which is the only property that makes it worth
running. A comment says to delete a deselect line if that test starts passing,
because a stale deselect silently hides coverage.

**Verified**: all 70 entry points in the repository respond to `--help`
without an import error (checked separately); the rewritten script now runs to
completion.

---

<a id="d-01"></a>
### D-01 -- A second stale decoding assertion, found only because V-01 widened coverage -- **CORRECTNESS** -- FIXED

Running the repaired validator immediately surfaced a failure the narrower
per-directory pytest command in `MACHINE_HANDOFF.md` never collected:

`inference_server/test_functional_planner.py::test_thinking_sampling_matches_checkpoint_guidance`

It asserted `temperature 1.0`, `presence_penalty 1.5`, `repetition_penalty 1.0`
for `qwen35-9b` -- the card's **general-task** thinking profile. The registry
serves the **precise-coding** profile (`0.6` / `0.0` / `1.05`), which is the
decoding decision recorded in `BASELINE_FIDELITY.md`, chosen precisely because
the general-task penalty made this checkpoint overrun on schema-constrained
output.

This is the same defect as the `llm3_baseline` one fixed before the audit, in a
different package: a test that hardcodes a copy of `models.json` and was not
updated when the registry moved. Both are now aligned, with a comment saying
the duplication is deliberate -- the test exists so a registry change cannot be
silent -- and that the two must be updated together.

**Why this one hid for so long:** `pytest.ini` lists `inference_server` as a
testpath, but the invocation in `MACHINE_HANDOFF.md` does not include it. The
documented "expected failures" count was measured with a command that never ran
this test. The handoff now says so explicitly.

---

<a id="f-01"></a>
### F-01 -- Nine of thirteen dispatcher failure codes had no reporting category -- **CORRECTNESS** -- FIXED

**What.** `mujoco_scenes.tamp.skills.FailureCode` defines thirteen codes the
shared `MuJoCoSkillDispatcher` can emit. `FAILURE_CATEGORIES` in
`make_paper_tables.py` recognised four of them:

```
covered: grasp_failed  placement_failed  precondition_failed  internal_error
gaps:    collision  ik_failed  path_blocked  target_occupied  object_not_visible
         no_candidate  function_unsatisfied  effect_not_observed  inference_failed
```

**Why it did not bite the Living Room grid.** That path runs through
`LivingRoomPhysicalExecutor`, which maps skill outcomes onto its own six-code
vocabulary in `baseline_common/living_room_execution.py`. All six were covered,
which is why the 180-episode grid categorises 81 of 81. The dispatcher's
thirteen codes are the **Kitchen** path, and Kitchen has no execution data yet.
So this was a live trap for the next grid rather than a correction to a
reported number.

**Fixed.** Mapped into the existing rows, using both spellings (lower-case code
and the upper-case terminal status `physical_terminal_status` derives from it):

| code | row | confidence |
|---|---|---|
| `ik_failed`, `collision`, `path_blocked` | IK or collision failure | certain -- the row is named for them |
| `effect_not_observed` | Final-state verification failure | certain |
| `target_occupied` | Geometric incompatibility | high |
| `no_candidate`, `object_not_visible` | No valid functional assignment | reasonable |
| `function_unsatisfied` | Geometric incompatibility | **judgement call, see below** |
| `inference_failed` | *excluded from the denominator* | see below |

**Two of those deserve a second opinion.**

`function_unsatisfied` was placed under geometric incompatibility on the
reading that a functional requirement the world cannot satisfy is a geometry
problem. It could equally be a final-state verification failure. Nothing has
emitted it yet, so nothing is misreported today, but it is worth confirming
before Kitchen runs.

`inference_failed` is now grouped with `MODEL_OUTPUT_TRUNCATED` as an
**infrastructure status**, excluded from the plan-rate denominator rather than
counted as a failure to plan. An unreachable model server is not a method's
inability to plan, and `vlm_tamp_baseline/run_living_room.py` already refuses
to write an execution artifact at all in that case -- this makes the reporting
side agree with that existing decision.

**Verified**: 13/13 dispatcher codes now categorise; the Living Room table is
byte-identical (81 of 81, same per-row counts).

---

<a id="p-01"></a>
### P-01 -- `BASELINE_FIDELITY.md` described a semantic-neutral condition; the visual baselines are given semantic aliases -- **DOC** -- **RESOLVED: the code is faithful, the prose was wrong. FIXED, no re-run.**

> **Resolution (2026-09-05).** Both source papers were read to settle this.
> **VLM-TAMP** (arXiv:2410.02193): Fig. 3 is "an example input image to the VLM,
> which are **annotated with object names** and bounding boxes", and Sec. III-A
> states "we provide the VLM **a text description of the scene that lists the
> objects** and the relations they satisfy ... generated by rendering the scene
> in the PyBullet simulator and **labeling the observable objects using
> ground-truth semantic segmentation**". Its example VLM responses are in
> semantic terms (`Opened(fridge door)`, `Picked(chicken leg)`).
> **OWL-TAMP** (arXiv:2411.08253v4): Sec. 3 defines a task over "a set of
> **named objects** (identified via object detection and segmentation)", and
> its sketches read `detach("place the banana near the apple and pear",
> banana, ...)`.
>
> Both originals give the model semantically named objects. Exposing aliases is
> therefore **faithful**, and removing them would have been a silent deviation
> from the methods being compared. Reading (2) below is wrong; reading (1) is
> correct. `BASELINE_FIDELITY.md` has been corrected in four places and now
> cites both papers. **No re-run.**
>
> The one caveat that survives is the retrieval comparison, now recorded in
> `BASELINE_FIDELITY.md`: the FM baselines are given the object-to-role mapping
> and the retrieval control must infer it, so the gap between them is not
> evidence that the FM solved grounding from pixels.

This is the most consequential finding in this audit. It does not change any
number, but it changes what the numbers *mean*, so it is written out in full.

**What the document says.** `BASELINE_FIDELITY.md` is the file that "defines
what may be claimed". It states, in four places:

> "a **semantic-neutral** textualized state ... It does **not** receive
> simulator semantic names or hidden inventory."
> "Five **ID-annotated** camera views replace the paper's semantically
> annotated PyBullet montage. MuJoCo instance segmentation supplies persistent
> ID correspondence **without class names**."
> "The original VLM-TAMP prompt included semantically named objects and
> annotated images. The shared-domain protocol **deliberately removes those
> labels** so that both visual baselines receive the same evidence."

**What the code and the artifacts actually do.** Both visual baselines call
`Observation.as_annotated_prompt_dict()`, which layers a `semantic_annotations`
block on top of the neutral dict (`baseline_common/models.py:197-210`). Verified
against the recorded requests of the 180-episode grid, not inferred:

- The alias map is in the **user message text** of the request actually sent:
  `"semantic_annotations":{"objects":[{"alias":"cup_1","id":"object_0001"},
  {"alias":"saucer_1",...},{"alias":"tv_remote",...}],"regions":[{"alias":
  "personal_table_left",...},{"alias":"shared_table",...}]}`
- The annotated PNGs sent alongside carry the same labels **drawn into the
  pixels**. `annotations.json` records
  `object_annotation: UNIQUE_SEMANTIC_ALIAS_ONLY`,
  `region_annotation: UNIQUE_SEMANTIC_ALIAS_ONLY`, `semantics_exposed: true`,
  and the rendered frame visibly reads "cup 1", "saucer 1", "tv remote",
  "personal table left", "shared table".

The goal string is *"place one cup and one saucer on each person's fixed
individual side table, and place the TV remote on the fixed shared coffee
table."* With that alias map, the object-to-role grounding is supplied rather
than inferred.

**The code is coherent and deliberate, not buggy.** `semantic_labels_exposed`
is an explicit named flag, set `True` on the vlm_tamp/owl_tamp paths and
`False` on llm3. `as_annotated_prompt_dict`'s own docstring says "Observable
state plus the aliases **printed on supplied RGB views**" -- the text mirrors
the pixels rather than adding to them. This is a designed condition that the
prose no longer describes.

**Fairness between the two FM baselines is intact.** Verified by scanning the
real prompts: VLM-TAMP 6/6 aliases present, OWL-TAMP 6/6. They receive
identical evidence, so the 93.3% vs 71.7% gap is *not* an information
asymmetry between them.

**But the third column is not like-for-like.** The retrieval baseline crops the
**raw, unannotated** frames on purpose -- `BASELINE_FIDELITY.md` explains why:
"the annotated frames carry printed semantic aliases, and CLIP reads text, so
scoring those would leak the answer through the pixels rather than testing
perception." So retrieval must ground from pixels while the FM baselines are
handed the mapping. Reporting 93.3% / 71.7% / 0% side by side without saying so
invites the reading that the FM does the grounding work, when the grounding was
given to it and withheld from the control.

**Why nothing was changed.** Two readings are possible and they lead to
opposite actions:

1. *The condition is intended and the prose is stale.* Then the fix is to
   correct four passages in `BASELINE_FIDELITY.md`, state plainly that the
   visual baselines receive semantic aliases in both pixels and text, and add
   the caveat to the results table. No re-run.
2. *The prose is the intent and the code drifted.* Then the visual baselines
   should call `as_semantic_neutral_prompt_dict()` with ID-only annotated
   frames, and **the entire 180-episode grid must be re-run**, because the
   model was given information the protocol says it must not have.

Choosing wrongly either misdescribes the experiment or discards 13 hours of
compute. This needs the researcher.

**Reproduce this check:**
```
.venv/bin/python -c "
import json,glob,pathlib
p=sorted(glob.glob('runs/living_room/execution/feasible_i9_20260904/vlm_tamp/L1/images_3/seed_000/model_calls/*.json'))[0]
print('cup_1' in json.dumps(json.loads(pathlib.Path(p).read_text())['model_requests']))"
```

---

<a id="a-01"></a>
### A-01 -- The prompt-leakage auditor exists and is never called -- **MISLEADING** -- **FIXED: now enforced at the shared transport**

`mujoco_scenes/functional_tamp_pipeline/audit.py` defines
`audit_prompt_leakage()`, which checks an outgoing request for forbidden
checker predicates (`OPEN_CAVITY`, `INSERTABLE_IN`, ...), canonical region
tokens (`D1`, `C2`, `B1`, ...) and oracle symbols (`GTSpecProvider`,
`KitchenGroundTruth`, ...). **Nothing in the repository calls it.** A written
guarantee that never executes is worth less than the paragraph describing it,
because a reader reasonably assumes it runs.

**Run retroactively over the completed grid**, against the request side only
(the function deliberately excludes model responses): **286 model-facing
request payloads audited, 0 containing any forbidden token.** So the guarantee
this auditor was written to enforce does hold for the reported run -- it simply
was not being checked.

Note this is a different and narrower question than [P-01](#p-01): the
forbidden lists cover internal checker predicates, canonical *storage* region
names and oracle class names. Semantic aliases such as `cup_1` are not on those
lists, and the auditor passes the payloads that contain them.

**Wired up (approved by the researcher, 2026-09-05).**
`baseline_common/inference.py` now audits every outgoing request in
`OpenAITransport.complete()` before it is sent, and raises `PromptLeakageError`
if a forbidden token is present. Because every model-driven method shares this
one transport, the property now holds by construction for all of them at once.

Design choices worth knowing:

- **`PromptLeakageError` is deliberately not a `PlanningError`.** The
  executives catch `PlanningError` to retry or to score a planning failure;
  leakage is neither. It means the harness was about to hand the model
  privileged evaluator information, and the right outcome is a loud abort, not
  a retried episode or a metric. Pinned by a test.
- **Audited before the request is sent**, so a contaminated prompt is never
  transmitted rather than merely noticed afterwards.
- **Image bytes are excluded from the scan.** Each request carries ~1.5 MB of
  base64 per camera, and the semantic aliases in those frames are drawn into
  the *pixels*, not written into the encoding -- so the base64 cannot leak a
  token as literal text, and scanning it would cost the grid hundreds of MB of
  string work for nothing. `_auditable_text` replaces data URLs with
  `<image>` and keeps everything else.
- **An unavailable auditor reports itself as unavailable** rather than
  silently passing, so a trace can never show a clean audit that never ran.

**Verified two ways.** Synthetic leaks of all three classes (canonical region
`C2`, oracle `GTSpecProvider`, checker `OPEN_CAVITY`) are refused, and a clean
prompt with a 500 KB image passes. Then **all 325 real VLM-TAMP requests from
the completed grid were replayed through the guard: 325 passed, 0 refused** --
so enabling it changes no completed episode and costs nothing on the healthy
path.

Six regression tests added in `baseline_common/tests/test_inference_errors.py`.

---

<a id="p-02"></a>
### P-02 -- `BASELINE_FIDELITY.md` forbade the exact claim the grid was run to make -- **DOC** -- RESOLVED, FIXED

**What.** The document that "defines what may be claimed" stated, for *both*
visual baselines:

> "The Living Room sequence experiment is **planning-only** ... do not move
> MuJoCo or count as physical execution. Its results therefore support plan/GT
> agreement claims only, **not execution-success claims**."
> "The implementation is planning-only and supports sequence-to-GT claims,
> **not physical execution-success claims**."

Meanwhile `baseline_common/run_baseline_execution_batch.py` passes
`--physical-execution` unconditionally, and every artifact in the
180-episode grid records `physical_execution: true` with real executed action
counts. The reported 93.3% / 71.7% / 0% are physical execution-success rates --
precisely what those clauses prohibited.

**Resolution.** Confirmed by the researcher on 2026-09-05: the clauses predate
this work. The physical execution layer landed in `a73cc89` and the
planning-only restriction was never lifted in the prose. **Execution-based is
the default condition from now on unless stated otherwise.**

**Fixed.** Both clauses now state that the Living Room experiment is physically
executed and that reported success is an execution-success claim, with a note
that `run_plan_gt_batch` remains the planning-only path and writes no execution
artifact. The OWL-TAMP entry additionally records that physical execution is a
**deliberate extension beyond that paper's simulation condition** -- the paper
reports plan feasibility, not physical task success on this domain -- so the
extension is described rather than silently assumed.

**Left unchanged deliberately:** the Workshop W1--W10 clause still says
planning-only, which is correct. Workshop grounding, sequencing and
manipulation are genuinely not implemented (`EXECUTION_AND_TESTING.md`), so
that restriction is a fact about the code, not stale prose.

**Pattern worth noting.** P-01 and P-02 are the same failure mode: the phase-4
port changed the experimental condition and `BASELINE_FIDELITY.md` was not
updated with it. Both were found by comparing the document against the recorded
artifacts rather than against the code. That comparison is cheap and is worth
repeating whenever the execution layer changes -- the artifacts record what
actually happened, and the document records what is allowed to be said about
it; when they disagree, one of them is wrong and it matters which.

---

<a id="p-03"></a>
### P-03 -- The stale planning-only description was repository-wide, including a module docstring contradicting its own flags -- **DOC** -- FIXED

Once [P-02](#p-02) was confirmed, the same claim was traced through every
document and docstring. It appeared in six places, all describing a condition
that stopped being true when the phase-4 execution layer landed:

| location | said | corrected to |
|---|---|---|
| `vlm_tamp_baseline/run_living_room.py` line 1 | "Run **planning-only** VLM-TAMP on one Living Room variant" -- while the module defines `--physical-execution` | execution is the default reported condition; without the flag no execution artifact is written |
| `owl_tamp_baseline/run_living_room.py` line 1 | "Run **planning-only** OWL-TAMP on one Living Room benchmark variant" | same, plus a note that physical execution extends beyond the paper's own simulation condition |
| `vlm_tamp_baseline/README.md` | "The Living Room condition **deliberately stops before physical execution**" and "annotates **only** persistent object/region IDs" | physically executed; annotates IDs **and** semantic aliases |
| `owl_tamp_baseline/README.md` | "The experiment remains planning-only and scores plan-to-GT sequence agreement" | Living Room executed; GT agreement produced alongside and not to be conflated |
| `README.md` | OWL-TAMP "supports **planning-only** Kitchen and Living Room" | Living Room physically executed by default |
| `CLAUDE_HANDOFF.md` | "Living Room: **planning-only** baseline adapter" | physically executed adapter, with the completed grid referenced |

The two module docstrings are the worst of these: they are the first thing a
reader sees, they contradict the module's own argument parser, and they would
lead someone to believe a completed execution grid was a planning experiment.

**Left unchanged, because they are still true:** the Workshop W1--W10 clauses
in both READMEs and `BASELINE_FIDELITY.md`, and the receding-horizon symbolic
executor description. Workshop manipulation is genuinely not implemented.

**One document was already right.** `owl_tamp_baseline/README.md` states that
the common input contains "the annotated RGB views, the goal, and the
observable **alias-to-ID map**" -- an accurate description of the alias
exposure that `BASELINE_FIDELITY.md` denied in [P-01](#p-01). When two documents
disagree, the artifacts settle it; here the more specific document was the
correct one.

---

<a id="n-01"></a>
### N-01 -- The fidelity document states the camera count as fixed at five; the reported grid used three -- **DOC** -- FIXED

`BASELINE_FIDELITY.md` said "five RGB views", "Five annotated camera views",
"anonymous IDs in the five views" and "the same five ID-annotated images" --
four statements of a fixed five-view input condition.

The view count is a **parameter** (`--camera-count`, validated to a nested
subset of 1/3/5 in `run_baseline_execution_batch.py:191`) and an intended
ablation dimension. Living Room defines five cameras (`l2_camera_close`,
`_front`, `_left`, `_right`, `_top`) and the runtime defaults to 5 -- but the
**2026-09-05 execution grid ran with 3**, confirmed from both the manifest and
the artifacts (`camera_count: 3`, three non-raw PNGs per observation).

Writing "five RGB views" in a paper whose reported grid used three would
misstate the condition. Corrected to describe it as the ablation parameter it
is, to name the value used, and to point at the per-episode `camera_count`
field rather than a number in prose.

**Every other numeric claim in the document was checked and is correct:**

| claim | code | |
|---|---|---|
| 12 diverse skeletons (VLM-TAMP) | `pddlstream_refiner.PAPER_MAX_SKELETONS = 12` | OK |
| at most five skeletons (OWL-TAMP) | `refinement.PAPER_MAX_SKELETONS = 5` | OK |
| 500 samples per action | `refinement.PAPER_MAX_SAMPLES_PER_ACTION = 500` | OK |
| `--max-sketch-actions` default 24 | argparse default 24 | OK |
| `--max-model-calls` 10 | argparse default 10 | OK |
| max output tokens 24576 | `models.json` planner budget | OK |

---

<a id="g-02"></a>
### G-02 -- Physical success is read from a dict, and that is sound -- **verified, not a defect** -- documented + test added

Worth recording because establishing it took reading three functions, and a
reader who stops after one would reasonably conclude the opposite.

`vlm_tamp_baseline/run_living_room.py` passes `goal_verifier=runtime.goal_verifier`
in **both** modes. `LivingRoomDiscoveryRuntime` (the physical runtime) does not
override it, so physical episodes are scored by the inherited verifier in
`LivingRoomPlanningRuntime`, which reads `self.locations` -- a plain dict. At
first glance that looks like physical success being decided symbolically.

**It is not.** `_place` assigns `self.locations[object_id] = region_id` only
after *all* of:

1. the pick/place controller returns without failure,
2. `_settle_payload` confirms the payload came to rest,
3. `_verify_place` confirms the physical ON relation -- support contact,
   footprint inside the observed support, payload non-overlap, no invalid
   environment penetration, settling drift within bounds, height consistency,
   and the 12-degree placement yaw tolerance.

Any of those failing returns `PLACE` and leaves `locations` untouched. The dict
is a *verified projection* of the simulator, so reporting its contents as
physical goal satisfaction is correct.

**Checked that the invariant actually holds**: `locations` is written in
exactly two places on the physical path -- `_pick` (to `None`) and `_place`
(post-verification). No wholesale reassignment anywhere.

**The one way it could break.** `LivingRoomSymbolicExecutor`
(`vlm_tamp_baseline/living_room_runtime.py:154,167`) writes `locations`
directly with **no** verification -- correctly, since it is the planning-only
path. Safety therefore rests entirely on one ternary in
`run_living_room.py:279-281` choosing `LivingRoomPhysicalExecutor` whenever
`--physical-execution` is set. If those two were ever crossed, every physical
success would become a symbolic claim and **no test would have caught it** --
`LivingRoomPhysicalExecutor` had no test coverage at all.

**Added** `baseline_common/tests/test_living_room_execution.py` (6 tests)
pinning the boundary: the executor must delegate every action through
`execute_phase2_action` rather than applying it, a rejected placement must
leave observable state untouched and stay recoverable, unsupported skills and
incomplete arguments must never reach the runtime, and a runtime missing the
contract must be refused at construction.

**Documented** the invariant on `LivingRoomDiscoveryRuntime`'s docstring,
including the explicit warning that anything writing `locations` outside
`_place` silently converts reported physical success back into a symbolic
claim. That docstring also drops the stale "Five-camera" description
(see [N-01](#n-01)).

---

<a id="p-04"></a>
### P-04 -- The annotation policy differs between Kitchen and Living Room and was documented nowhere -- **DOC** -- FIXED

Following [P-01](#p-01) through the rest of the tree turned up that alias
exposure is **not uniform across scenes**:

| scene | `object_annotation` | `region_annotation` |
|---|---|---|
| Kitchen (`baseline_kitchen_runtime.py:1119`) | `UNIQUE_SEMANTIC_ALIAS_ONLY` | `PERSISTENT_REGION_ID_ONLY` |
| Living Room (verified in grid artifacts) | `UNIQUE_SEMANTIC_ALIAS_ONLY` | `UNIQUE_SEMANTIC_ALIAS_ONLY` |

**The difference is correct and deliberate**, which is why it is documented
rather than changed. Kitchen's regions *are* its search problem: `D1`, `D2`,
`B1`, `C1`, `C2` are closed storage whose contents stay hidden until inspected,
and those exact tokens appear in `FORBIDDEN_CANONICAL_REGION_TOKENS` in the
leakage auditor. Naming them in a prompt would tell the model where to look.
Living Room has no closed regions, and its goal already names the roles ("each
person's fixed individual side table", "the fixed shared coffee table"), so
region aliases add nothing the instruction does not already give.

Nothing stated this anywhere. "The model was given semantic aliases" therefore
means something materially different in each scene, and a Kitchen result and a
Living Room result are not describing the same input condition. Added a table
and the rationale to `BASELINE_FIDELITY.md`, with an instruction to name the
scene when reporting.

**Also corrected**: `mujoco_scenes/DISCOVERY_REPLANNING.md` said the Living
Room runner uses "the same five fixed cameras, persistent **anonymous** IDs" --
wrong on both counts, the same drift as [N-01](#n-01) and [P-01](#p-01), and on
the path of the *proposed framework* rather than a baseline. Fixed.

**Not changed, flagged instead**: the framework's Living Room runner defaults
to `camera_count=5` while the completed baseline grid ran at 3. If the
framework is run at its default, its column will not share the baselines'
input condition. Pass `--camera-count 3` to match, or re-run the baselines at
5 -- but do not compare across the two.

---

<a id="v-02"></a>
### V-02 -- Integrity sweeps over configs, variant labels and scene assets -- **all clean**

Recorded because a later reader may run the same checks and needs to know the
expected answer, and because two of these produced false positives that are
easy to repeat.

| sweep | result |
|---|---|
| 58 YAML configs parse | clean |
| 162 tracked JSON files parse | clean |
| 6 scene XMLs parse | clean |
| 43 `object_library.xml` asset keys resolve | **43/43** |
| Living Room variant labels: docs vs YAML vs code | **10/10 agree** |
| Kitchen variant labels | 12/12 present |
| `EXPECTED_GT_ACTIONS/` coverage | 10 / 12 / 10, complete |
| docstrings citing numbers absent from their own body | 0 |
| modules never referenced anywhere | 0 |
| structurally identical function bodies across modules | 0 |

**Two false positives worth not repeating.**

*"105 missing scene assets."* MuJoCo scenes here are built with
`MjModel.from_xml_string(xml_str, assets=...)`, so a `file="ycb/mug/ycb_mug.obj"`
attribute is a **key into an in-memory asset dict**, not a path relative to the
XML. Checking the filesystem next to the XML reports every asset missing.
Resolve against `mujoco_scenes/assets/objects/meshes/` instead: 43/43 present.

*"EXPECTED_GT_ACTIONS is missing every variant."* That tree is keyed by
**paper labels** (`L1`..`L10`, `K1`..`K12`, `W1`..`W10`), not by internal names
(`F0_ALL_OBJECTS_IN_STAGING`, ...). Comparing it against `VARIANT_LABELS`
values rather than positions reports a total mismatch when the coverage is in
fact complete.

**The variant mapping is sound**, which is worth stating given the trap
recorded in `CLAUDE_HANDOFF.md` ("`[L2RegionScene]` and the `L2_` scene prefix
are not variant L2"). The trap is real -- scene *prefixes* are unrelated to
paper labels -- but the label mapping itself agrees three ways: the catalogue
document, `living_room_variants.yaml`, and
`final_paper_variant_labels.paper_variant_label()` all resolve L1 to `F0_...`
through L10 to `I3_NO_TABLES` with no discrepancy.

---

<a id="k-01"></a>
### K-01 -- The Kitchen serving allocator cannot place the benchmark's own four serving targets above a 6.5 cm footprint -- **CORRECTNESS, blocks Kitchen execution** -- **DECISION REQUIRED, nothing changed**

Three of the five remaining known-failing tests are the serving allocator, and
they have been carried as "unfinished Kitchen work" since the phase-4 port
without a diagnosis. This is the diagnosis, and it is worse than a stale test.

**The rule.** `KitchenPlacementResolver.footprints_overlap` treats two payloads
as conflicting when their centres are closer than
`(w_a + w_b)/2 + 0.012` in *both* axes -- a 12 mm clearance margin.

**The canonical layout violates that rule.** `fixed_serving_slots` encodes what
the comment calls "K1/K2's successful layout ... the canonical layout for every
variant":

```
object_0001 (coffee) (-0.15, -0.48)      object_0002 (coffee) ( 0.15, -0.48)
object_0003 (soup)   (-0.08, -0.54)      object_0004 (soup)   ( 0.08, -0.54)
```

Coffee-to-soup separation is dx = 0.07, dy = 0.06. For any square footprint `s`
the pair is rejected unless `0.07 >= s + 0.012` or `0.06 >= s + 0.012`, i.e.
**`s <= 5.8 cm`**. So the allocator rejects its own canonical slots for
anything larger, falls back, and runs out of candidates.

**Measured, by sweeping the footprint:**

| square footprint | serving targets placed |
|---|---|
| 4.0 -- 6.5 cm | **4 / 4** |
| **7.0 cm and above** | **3 / 4** |

**`footprint()` returns `(0.10, 0.10)` when perception supplies no measured
dimensions** -- the documented "conservative execution fallback". At 10 cm the
allocator places three of four. The Kitchen goal is "coffee and soup for **two**
people", so four serving targets is the benchmark's actual requirement, not an
edge case. Cups and bowls are realistically 8--12 cm.

**Why the tests fail, precisely:**

- `test_serving_allocator_is_deterministic_and_role_separated` asserts the first
  soup bowl lands at x = -0.15. The code assigns soup `slot_order = (-0.10,
  0.10, 0.0)`, deliberately, with the recorded reason "Match the physically
  successful final two-bowl layout even when the second bowl is retrieved and
  served first." That one **is** a stale assertion -- code and test came from
  different sides of the two-way merge in `a73cc89`.
- The other two fail with `DESTINATION_RESOLUTION_FAILED: serving support is
  full` on the **third** payload, which is the capacity limit above, not a
  stale assertion.

**Why this was invisible.** Kitchen has no execution data, so nothing exercised
the allocator end to end. The Living Room grid never touches this code.

**Why nothing was changed.** Three defensible fixes exist and they are not
equivalent:

1. **Widen the canonical slots** so they satisfy the 12 mm rule. Changes a
   layout the comment says was physically validated -- the geometry may have
   been chosen to keep the coffee-serving corridor clear for later retrieval,
   which the clearance rule knows nothing about.
2. **Reduce the clearance** below 12 mm, or apply it only between payloads that
   are not in the fixed table. Relaxes a physical validity criterion, which is
   the same class of change as the self-collision widening and needs the same
   scrutiny.
3. **Widen the candidate set** so the allocator can find non-grid positions
   instead of failing. The support has room -- x is usable across
   +/-0.167 -- so three 10 cm payloads *do* fit at roughly -0.112 / 0 / +0.112;
   the greedy fixed-slot-first order is what strands the third.

Option 3 changes no physical criterion and no validated layout, so it is the
one I would propose -- but all three alter Kitchen execution behaviour, and the
right choice depends on why those exact canonical coordinates were chosen,
which is not recorded anywhere I can find.

**This should be settled as part of the Kitchen/Workshop execution port**
(confirmed upcoming, 2026-09-05), not after a grid has been attempted: as it
stands the serving task cannot complete with realistically-sized payloads, and
the failure surfaces as `DESTINATION_RESOLUTION_FAILED` on the *fourth* target,
which is late enough in an episode to look like a planning fault rather than an
allocator capacity limit.

**Reproduce:**
```
.venv/bin/python - <<'PY'
from unittest.mock import Mock
from mujoco_scenes.kitchen_object_manipulation import KitchenPlacementResolver
objects=[{"generic_object_id":o,"selected_functions":[f],
          "observed_centroid_world_m":[x,0.,0.63],
          "observed_dimensions_m":{"length":0.10,"width":0.10}}
         for o,f,x in (("object_0001","coffee_vessel",-.2),("object_0002","coffee_vessel",.2),
                       ("object_0003","soup_bowl",-.1),("object_0004","soup_bowl",.1))]
res={"accepted":[{"generic_object_id":r["generic_object_id"],
                  "physical_backend_body":r["generic_object_id"]} for r in objects]}
r=KitchenPlacementResolver(Mock(), {"objects":objects}, res)
for o in ("object_0001","object_0002","object_0003","object_0004"):
    try:
        t=r.resolve(o,"serving_area"); r.record_successful_serving_placement(o,t)
        print(o,"->",t.target_position_world_m[:2])
    except ValueError as e: print(o,"-> FAILED:",e)
PY
```

---

<a id="i-01"></a>
### I-01 -- The repository has a deliberate integrity-guard culture, and the two gaps found were the two places it did not reach -- **context, no defect**

Recorded because it changes how the rest of this audit should be read, and
because it tells the next person what is already protected.

**Roughly 45 tests exist for the sole purpose of guarding information
boundaries.** A sample of what they enforce:

- `test_reference_evaluator_is_strictly_offline` -- the GT reference evaluator
  is never imported by any runtime module (enforced twice, independently)
- `test_new_pipeline_does_not_import_oracle_solution_generators`
- `test_zero_leakage_in_kitchen_and_workshop_payloads`
- `test_adapter_outgoing_payload_has_zero_checker_and_region_leaks`
- `test_production_observed_instances_leak_no_semantics`
- `test_production_candidate_regions_leak_no_ground_truth`
- `test_production_module_does_not_import_oracle_or_planning`
- `test_benchmark_runs_production_before_privileged_oracle`
- `test_unknown_relation_never_becomes_valid_compatibility_edge`
- `test_source_context_ignores_legacy_privileged_region_key`
- `test_folder_boundaries` -- the baselines cannot import each other, and the
  neutral layer cannot import either

**The GT/VLM specification boundary is clean.** `GTSpecProvider` is constructed
in exactly three places: the explicit `gt` mode in `provider_for_mode`, the
oracle-evidence ablation, and the offline reference evaluator. The VLM provider
contains no reference to it beyond one comment. The import is lazy, so a VLM
run does not even load the GT module.

**Why this matters for the rest of this document.** The defects found here are
not the result of carelessness about privileged information -- the opposite is
true, and the guards are unusually thorough for a research codebase. The two
real gaps were both at *seams* the guard culture had not been extended across:

1. **The shared baseline transport** ([A-01](#a-01)). Leakage was tested on the
   functional pipeline's adapter payloads
   (`test_zero_leakage_in_kitchen_and_workshop_payloads`,
   `test_adapter_outgoing_payload_has_zero_checker_and_region_leaks`) but never
   on `baseline_common`'s `OpenAITransport`, which is what the three comparison
   baselines actually send through. Now enforced there too, so the guarantee
   covers both paths rather than one.
2. **The physical/symbolic executor selection** ([G-02](#g-02)).
   `LivingRoomPhysicalExecutor` had no test at all, so the single ternary
   deciding whether reported physical success is real was unguarded. Now
   covered.

Both were seams between subsystems rather than holes inside one, which is the
usual place for this class of gap.


---

## K-02 — Serving placements were never recorded on the verified-release path

**Files.** [`mujoco_scenes/kitchen_ground_truth_execution.py`](mujoco_scenes/kitchen_ground_truth_execution.py),
[`mujoco_scenes/kitchen_object_manipulation.py`](mujoco_scenes/kitchen_object_manipulation.py)

**Severity.** Result-affecting. Fixed 2026-09-06.

### What was wrong

A successful Kitchen `PLACE(object, serving_area)` can finish through three
different routes in the ground-truth executor:

| Route | Returned status | Updated `inventory_by_id[...]["location"]` | Recorded in `serving_placements` |
|---|---|---|---|
| `KitchenObjectManipulationExecutor.place` | `PLACE_SUCCESS` | n/a (resolver-side) | yes, `kitchen_object_manipulation.py:4659` |
| `_execute_controlled_placement` recovery | `PLACEMENT_COMPLETED` | yes | yes |
| **verified-release fallback** | `RELEASED_PLACEMENT_VERIFIED` | **no** | **no** |

The third route confirms a release that completed while the gripper opened
along a path the resolver had not planned. It called
`validate_stable_placement`, saw the object was correctly and stably placed,
returned `success: True` — and then performed **only** the `countertop`
bookkeeping. For `serving_area` it did nothing at all.

Everything downstream that asks "what is already on the serving surface?" reads
`KitchenPlacementResolver.serving_placements`. There are five such consumers:

- `_allow_served_payloads_for_next_motion` (`kitchen_ground_truth_execution.py:414`)
- the bowl-descent contact allowance (`kitchen_ground_truth_execution.py:1755`)
- the serving-utensil stance search allow-list (`kitchen_ground_truth_execution.py:2196`)
- the serving-utensil `_local_stance` exemption (`kitchen_ground_truth_execution.py:2360`)
- **serving slot allocation itself** — the `footprints_overlap` guard in
  `KitchenPlacementResolver` (`kitchen_object_manipulation.py:1704`)

### How it presented

In `F5_FULL_DISTRIBUTED_SEARCH` four objects reach the serving area (actions
21, 23, 25, 29). Three of them — `ab3_narrow_deep_cup`, `ab3_medium_deep_mug`,
`ab3_deep_bowl` — completed through the verified-release fallback and were
never recorded. At the final action the resolver believed exactly one object
was on the surface:

```
allowed_contact=['ab3_shallow_bowl'], served_placements=['ab3_shallow_bowl']
```

So the allow-list that exists specifically to let the arm reach past an
already-served vessel did not contain `ab3_deep_bowl`, and seating the last
utensil was rejected on a 3–6 mm forearm overlap with it:

```
o1:COLLISION(approach=environment collision
  google:link_forearm / ab3_deep_bowl [ab3_deep_bowl_wall_5] (-0.3 cm),
  release=... [ab3_deep_bowl_wall_4] (-0.6 cm))
```

Every orientation candidate was exhausted and the episode failed with
`SERVING_UTENSIL_RELEASE_FAILED`.

### The second, quieter consequence

**This is the part that affects results beyond one variant.** Serving *slot
allocation* also consults `serving_placements`, through the `footprints_overlap`
rejection in the candidate loop. An unrecorded object is an invisible one, so
the allocator was free to hand out a slot that physically overlaps a vessel
already sitting there. It did not produce a visible failure in the runs
observed, because the fixed four-object layout happens to keep the semantic
slots apart — but the guard that is supposed to prevent an overlapping
allocation was inoperative for any object placed through this route, which was
most of them.

### The fix

1. `KitchenPlacementResolver.record_observed_serving_placement(object_id)` —
   records a serving placement from the payload's **actual resting pose** read
   back from the simulation. The existing
   `record_successful_serving_placement` requires a commanded `PlacementTarget`,
   which this route does not have. Reading the pose back is also the more
   accurate footprint centre, since it is where the object ended up rather than
   where it was aimed.
2. `KitchenGroundTruthExecutor.mark_object_served(object_id)` — collects the
   inventory-row update and the resolver call in one place, so a fourth success
   route cannot silently diverge the way this one did. The verified-release
   fallback now calls it for `serving_area`.

### Why the bookkeeping was not simply read from the inventory

An earlier attempt built the allow-list from
`inventory_by_id[object_id]["location"] == "serving_area"`. That is wrong for a
different reason: `location` is where Phase 1 *observed* the object, and two of
the three success routes never rewrite it either. The resolver is the right
source of truth; it just was not being written to.

### Effect

`F5_FULL_DISTRIBUTED_SEARCH` under `--strict-robot-execution`: 31/31 actions,
`SUCCESS`. Previously 30/31 with the final `PLACE_SERVING_UTENSIL` failing.

### Not fixed here

The three routes still exist and still return three different success statuses
for the same outcome. Consolidating them is a larger change to the placement
control flow and is not attempted while execution results are being collected.
`mark_object_served` is the seam that keeps them consistent in the meantime.

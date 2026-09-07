# Baseline fidelity and reporting protocol

This document defines what may be claimed for the paper baselines. The
implementations are algorithm-faithful ports evaluated in a shared new domain;
they are not exact replications of the original papers' robot, simulator,
tasks, or model.

## Frozen upstream references

| Method | Primary paper | Official code used for the audit | Frozen revision |
|---|---|---|---|
| VLM-TAMP | arXiv:2410.02193 | `Learning-and-Intelligent-Systems/kitchen-worlds` and its PDDLStream submodule | kitchen-worlds `1839f5ff4c41f6a6b0cf5abbeb7a1292b1a551c4`; PDDLStream `b38137e47fd4a4116a3e36bc4be691cbe5da6cb0` |
| LLM3 | arXiv:2403.11552 | `AssassinWS/LLM-TAMP` | `aca6f0c1ed5f7319b48b44523e4b317a15b3861f` |
| OWL-TAMP | arXiv:2411.08253v4 | No public author code release found; independent implementation from the paper | Not applicable |
| ViLaIn-TAMP | arXiv:2506.03270 | `omron-sinicx/ViLaIn` (earlier official ViLaIn release); ViLaIn-TAMP itself has no public release | Independent implementation from the paper |

The PDDLStream source is installed outside the repository by
`vlm_tamp_baseline/setup_pddlstream.sh`; the adapter checks its Git revision at
runtime. Small Python 3.11 and logging shims are applied without modifying the
pinned source.

## The task instruction is identical for every method

Each domain issues exactly one instruction, and the proposed method and every
baseline receive that same string.  It is defined once, in the scene
configuration each runtime already reads
(`kitchen_feasibility_variants.yaml:goal_instruction`,
`living_room_variants.yaml:task`,
`workshop_variants.yaml:canonical_task_instruction`), and re-exported by
`mujoco_scenes/benchmark_task_instructions.py`.  Nothing restates it.

| Domain | Instruction |
|---|---|
| Kitchen | Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil. |
| Living Room | Place one cup and one saucer on each of two fixed personal side tables and place the TV remote on the fixed shared coffee table. |
| Workshop | Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench. |

**This was not previously true, and results produced before 2026-09-07 are not
comparable across methods.** The text had been written out separately in the
batch runner, three scene configs, the pipeline's domain definitions and
several baselines, and the copies had drifted. Two differences mattered:

- The Workshop instruction given to the baselines named the object categories
  and prescribed the insertion geometry -- "the compatible **screw**", "the
  first compatible **driver**", "**tip-down**", "**head/recess on top**" --
  where the published instruction says only "the compatible **components**
  required to complete the fastening" and "reusable **equipment**". That is a
  substantial information difference in the baselines' favour: it supplies the
  category identification the task is meant to require.
- The Kitchen instruction given to the baselines added "Search the closed
  kitchen storage for anything still required", a procedural hint absent from
  the published instruction.

Both inflated the baselines relative to the proposed method. Any table mixing
pre- and post-2026-09-07 runs would understate the proposed method's margin
for a reason that has nothing to do with either method.

## VLM-TAMP correspondence

Preserved algorithmically:

1. The first VLM query produces ordered intermediate goals in English.
2. A second VLM query translates those goals to grounded formal predicates.
3. The shared-domain adaptation receives the open goal, the configured RGB
   views (`--camera-count`, a nested subset of 1/3/5; the 2026-09-05 grid used
   **3**, and every episode records its own `camera_count`)
   annotated with persistent instance/region IDs **and unique semantic
   aliases**, a textualized state carrying the same aliases, executed history,
   and the last failed subgoal. It does not receive hidden inventory, expected
   GT, checker predicates, canonical storage-region names, or oracle symbols.
4. Formal subgoals are solved sequentially by PDDLStream.
5. Each subgoal gets three TAMP attempts using goal-related, visible, then all
   manipulable objects, with at most 12 diverse skeletons per attempt.
6. Failed grounding, stream refinement, physical motion, or effect validation
   triggers a new two-stage VLM query.
7. This baseline's own published decoding is temperature `0.2` with thinking
   disabled, available as `--decoding paper`. It is not the default; see
   "Decoding conditions".

Embodiment adaptations:

- PyBullet/PR2 is replaced by the shared MuJoCo/Google-robot scene and skills.
- The PDDL domain and streams represent this kitchen's `INSPECT`, `PICK`,
  `PLACE`, `POUR`, and `STIR` operations.
- RRT* base paths and placement poses are sampled before execution. Pick/task
  geometry uses measured inventory geometry; final arm IK, collision, contact,
  and POUR/STIR trajectory validation remains lazy and is performed by the
  live MuJoCo skill. A failure is returned through the paper's reprompt loop.
- Annotated camera views replace the paper's semantically annotated PyBullet
  montage. The view count is a reported ablation dimension, not a fixed
  property of the protocol: 1, 3 or 5 nested fixed subsets, recorded per
  episode as `camera_count`. Living Room defines five (`l2_camera_close`,
  `_front`, `_left`, `_right`, `_top`); the reported execution grid used three. MuJoCo instance segmentation supplies both persistent ID
  correspondence and the semantic alias drawn on each box
  (`object_annotation: UNIQUE_SEMANTIC_ALIAS_ONLY`); this is an oracle
  instance tracker *and* an oracle labeller, and must be reported as such.
  The original does the same thing -- VLM-TAMP Fig. 3 is "annotated with
  object names", generated by "labeling the observable objects using
  ground-truth semantic segmentation" -- so this is a faithful port of the
  input condition, not a relaxation of it.
- Qwen3.5-9B may replace GPT-4o-mini. Report the model substitution explicitly.
- **The Living Room experiment is physically executed.** `--physical-execution`
  drives the calibrated Google-robot skills through the shared Living Room
  runtime, and `benchmark_execution_result.json` records physical goal
  satisfaction verified on the final simulator state. Reported success is
  therefore an execution-success claim, not a plan/GT agreement claim.
  (Before the phase-4 port this experiment was planning-only and this clause
  said so; the physical layer landed in `a73cc89` and the restriction no
  longer applies. Planning-only Living Room results, if any are still wanted,
  come from `run_plan_gt_batch`, which writes no execution artifact.)
- The Workshop W1--W10 experiment is also planning-only. It uses a separate
  symbolic PDDLStream domain for `INSPECT`, `PICK`, `PLACE`, `INSERT`, and
  `FASTEN`; no continuous stream or physical skill is claimed for this
  condition. Storage contents remain hidden from the VLM until a successful
  symbolic inspect update. The default one-call condition is initial-plan
  comparison; any higher call limit is reported as a symbolic-reprompt
  ablation.

The original VLM-TAMP prompt included semantically named objects and annotated
images, and **so does this port**: both visual baselines receive unique
semantic aliases in the annotated frames and in the textualized state, and they
receive *identical* exposure -- verified over the 2026-09-05 grid, 6/6 aliases
present in both VLM-TAMP's `model_requests` and OWL-TAMP's `model_prompts`.
This matches both source papers: VLM-TAMP provides "a text description of the
scene that lists the objects" over images "annotated with object names"
(arXiv:2410.02193 Fig. 3, Sec. III-A), and OWL-TAMP assumes "a set of named
objects (identified via object detection and segmentation)" (arXiv:2411.08253
Sec. 3). Removing the aliases would make these ports *less* faithful, not more.

Results from this repository remain an adapted protocol rather than an exact
input-level replication, because the robot, simulator, task suite and model all
differ -- not because the labels were withheld.

**This has a direct consequence for how the results table must be read.** The
two model-driven baselines are told which instance is a cup and which support
is the shared table; the retrieval baseline is not, because it scores crops
from the *raw* frames on purpose (see "Retrieval baseline correspondence": the
annotated frames carry printed aliases, and CLIP reads text). VLM-TAMP versus
OWL-TAMP is therefore a like-for-like comparison, but neither is a like-for-like
comparison with retrieval on *grounding*: they are given the object-to-role
mapping and retrieval must infer it. Report the retrieval column as an
open-vocabulary grounding control, and do not describe the gap between it and
the FM baselines as though the FM baselines had solved the grounding problem
from pixels.

### Annotation policy differs by scene, on purpose

The alias exposure described above is not uniform across domains, and the
difference is deliberate:

| scene | `object_annotation` | `region_annotation` |
|---|---|---|
| Kitchen | `UNIQUE_SEMANTIC_ALIAS_ONLY` | `PERSISTENT_REGION_ID_ONLY` |
| Living Room | `UNIQUE_SEMANTIC_ALIAS_ONLY` | `UNIQUE_SEMANTIC_ALIAS_ONLY` |

Kitchen withholds region names because its regions *are* the search problem:
`D1`, `D2`, `B1`, `C1` and `C2` are closed storage whose contents are hidden
until inspected, and those exact tokens are on the forbidden list in
`mujoco_scenes/functional_tamp_pipeline/audit.py`. Naming them would tell the
model where to look.

Living Room has no closed or hidden regions, and its goal already names the
roles ("each person's fixed individual side table", "the fixed shared coffee
table"), so region aliases reveal nothing the instruction does not already
state. The `object_annotation` mode is identical in both.

Record the scene when reporting, because "the model was given semantic aliases"
means something different in each.

The old direct subgoal-to-action templates are not the baseline. They are
available only as `--refiner catalog-ablation`.

## OWL-TAMP correspondence

Preserved algorithmically:

1. Relaxed grounding enumerates reachable discrete actions while deferring
   continuous parameters optimistically.
2. A five-image VLM query produces a partial discrete sketch and goal facts.
3. `Executed(i)` preconditions/effects constrain the symbolic solution to
   contain that sketch as an ordered subsequence.
4. Separate VLM calls translate each sketched action's physical requirement
   into a restricted geometric constraint expression.
5. Refinement uses discrete search followed by bounded continuous sampling:
   500 samples per action and at most five skeletons.
6. The paper's simulation condition is single-shot; it does not inherit
   VLM-TAMP's reprompt loop. The paper's real-robot appendix separately
   describes a receding-horizon observe--plan--execute policy.

Domain adaptations and reporting restrictions:

- The paper's simulation domains are replaced by the shared MuJoCo Kitchen and
  Living Room schemas. The model receives the same annotated views and
  alias-carrying textualized state as VLM-TAMP -- identical exposure, verified
  over the 2026-09-05 grid -- which is what makes the two columns comparable.
- **The Living Room implementation is physically executed**, through the same
  shared runtime and the same goal verifier as every other method, so its
  reported success is an execution-success claim. This is a deliberate
  extension beyond the paper's simulation condition and must be described as
  such: the paper reports plan feasibility, not physical task success on this
  domain. `--protocol receding_horizon` remains a separately named symbolic
  observe--plan--apply-one-action condition and is still not a physical
  reproduction of the paper's real-robot deployment.
- Receding-horizon episodes apply one action per planning cycle, so
  `--max-replans` caps executed actions at `--max-replans + 1`. Episodes that
  end in `REPLAN_BUDGET_EXHAUSTED` under a budget smaller than the expected
  action count are budget-terminated and must not be reported as the method
  failing the task.
- Kitchen closed-storage contents are not provided. Hidden-object variants
  expose the single-shot baseline's partial-observability limitation; automatic
  inspection/replanning would be a separately named condition.
- Workshop storage contents follow the same rule. The VLM can choose the
  visible `INSPECT` action, but the single-shot condition does not automatically
  inspect storage or reveal its contents.
- No YOLO/SAM output, functional ranking, proposed search module, expected GT,
  hidden inventory, simulator names, or functional assignment is model input.
- The native protocol issues one constraint request per sketch action, so
  `--max-sketch-actions` (default 24) bounds that cost; a sketch that
  degenerates into repetition would otherwise bill one request per repeat.
  Truncation is recorded as `constraint_generation_complete` in
  `model_trace.json`.
- Because the authors have not released code, call this an “OWL-TAMP
  paper-derived reimplementation,” never an official code port or exact
  replication. Keep `model_trace.json` and the method manifest for audit.

For K1--K12 planning-only trials, both visual baselines construct the variant
directly and use MuJoCo instance segmentation solely to assign persistent
anonymous IDs in the configured views. They do not consume the proposed framework's
Phase-1 object registry or functional witness. GT/backend ID translation occurs
only after planning. Report raw execution-vocabulary agreement and the shared
task-vocabulary normalization separately.

## Retrieval baseline correspondence

This is not a port of a published TAMP system. It is a deliberately minimal
open-vocabulary grounding baseline that isolates how far similarity-based
retrieval alone gets on the benchmark, with no language model in the loop.

1. The task structure is a fixed role template rather than a planned
   decomposition: two personal supports, one shared support, two drink
   vessels, two under-dishes, one handheld control.
2. Each role is filled by CLIP ViT-B/32 image-text similarity between the
   role's function phrase and a crop taken from the raw, unannotated frames.
3. Role phrases name functions, never category nouns, so the baseline is not
   handed the answer inside its own query.
4. Candidate supports are restricted to the runtime's registered support
   regions. The staging area is where payloads start and is not a placement
   target; electing it would let a missing table masquerade as a usable one.
5. A role that cannot be filled from distinct candidates yields
   `NO_RETRIEVED_ROLE_FILLER` and an infeasible verdict.
6. `uses_language_model` is false and every request counter is structurally
   zero, so this baseline is unaffected by the decoding conditions below and
   its numbers do not move with the served checkpoint.

Physical execution runs the retrieved assignment through the same shared
Living Room skills as the other methods, so a wrong grounding appears as a
physically executed wrong plan rather than a planning-only mismatch. Because
the physical runtime re-settles the scene before observing, its crops are not
bit-identical to the planning path's; render size is held equal across both so
the difference is scene settling only.

Call this an "open-vocabulary retrieval baseline", never a TAMP baseline.

## LLM3 correspondence

Preserved algorithmically:

1. The model returns a full plan from the current state, not a one-action
   policy or functional decomposition.
2. Every primitive contains discrete arguments and continuous parameters.
3. Motion execution returns success/failure feedback for the attempted plan.
4. The last three plan traces are returned to the model.
5. The model may resample failed continuous parameters or symbolically
   backtrack by changing the full plan.
6. The original decoding condition is temperature `0`, thinking disabled.
   Greedy decoding is not used on this checkpoint; see "Decoding conditions".

Embodiment adaptations:

- The original GPT-4/PyBullet 2-D box arrangement domain is replaced by a
  frozen VLM and the common MuJoCo kitchen.
- Original `place(x,y,theta)` becomes bounded placement offsets/yaw. POUR and
  STIR add scene-relevant continuous values. These values are consumed by the
  physical motion layer; they are not discarded after prompting.
- The original LLM3 release is text-only. In this visual-domain adaptation,
  its textual state is restricted to persistent IDs and observable relations,
  while the same annotated images provide semantic evidence. Report
  this modality change explicitly.

Do not describe this as the original LLM3 benchmark. Use “LLM3 algorithm port
to the shared kitchen domain” or “LLM3-style task-and-motion baseline.”

## Decoding conditions

Two conditions exist and are reported separately. Every method in a table must
run the same one; the batch runners take `--decoding` and record it in
`protocol_manifest.json`.

`model-native` is the default and the main reported condition: thinking
enabled, with Qwen3.5-9B's published thinking-mode sampling for precise coding.

| Parameter | Value | Source |
|---|---|---|
| `temperature` | 0.6 | Qwen3.5-9B card, thinking / precise coding |
| `top_p` | 0.95 | same |
| `top_k` | 20 | same |
| `min_p` | 0.0 | same |
| `presence_penalty` | 0.0 | same |
| `repetition_penalty` | 1.05 | deviation, justified below |
| max output tokens | 24576 | card's planner budget |

The published figures are used unchanged except `repetition_penalty`, which is
the single documented deviation. The card's other thinking profile, for general
tasks, sets `presence_penalty` 1.5 with no repetition guard; on
schema-constrained plan output that made this checkpoint run past its stopping
point. An OWL-TAMP Living Room discrete sketch reached 18 to 64 actions for a
10-action task and emitted mutually contradictory goal literals, and the two
penalties applied together truncated the plan to 4 actions instead.
`repetition_penalty` 1.05 is the smallest value that held the sketch to its
correct length: 1.03 still degenerated to the 64-action parser cap, while 1.05
and 1.10 each returned exactly the 10 correct actions across repeated samples.
Qwen's own guidance warns this checkpoint produces endless repetitions without
a repetition guard and advises against greedy decoding.

`paper` reproduces each baseline's own published condition, temperature 0.2
with `top_p` 1.0 and thinking disabled, selected by `--decoding paper`. That
condition is near-greedy, and it is where the degeneration above was first
observed, so it is reported as an ablation rather than as the headline result.

Thinking is enabled for both visual baselines and is not a free variable. With
thinking disabled, VLM-TAMP's first Living Room subgoal list asked the
one-gripper robot to hold two objects at once and the episode failed; with
thinking enabled the same variant was solved on the first model call. A
per-method difference in thinking would confound the comparison, so the two
baselines must share the flag.

## ViLaIn-TAMP correspondence

Preserved algorithmically:

1. Four independent model calls: object estimation, initial-state estimation,
   goal-state estimation, and Corrective Planning. The architectural and
   prompt boundaries between them are kept separate.
2. The estimated state is compiled to PDDL and solved by **Fast Downward
   24.06.1** with `lama-first`, the paper's symbolic planner and search.
3. Generated plans are validated with **VAL**; an invalid or unplannable
   problem enters Corrective Planning, bounded by `--cp-limit` (0-3).
4. Geometric refinement precedes execution.

Domain adaptations and reporting restrictions:

- **Qwen-only reproduction.** The paper allocates object estimation to a local
  Qwen2.5-VL checkpoint and reasoning to a `gpt-4o-2024-08-06` snapshot. This
  runs all four calls against the served Qwen3.5-9B, so it is a Qwen-only
  reproduction and not the paper's model allocation. Report it as such.
- Geometric refinement is a MuJoCo cloned-scene sequence preflight, an
  adaptation of the paper's MoveIt Task Constructor refinement, not a
  reproduction of it.
- The baseline owns its observations, PDDL problems, plans, refinement records
  and execution projections. It does not consume `G_F`, `G_O`,
  `ground_graph()`, `phi*`, or any of their assignments or witnesses.
- It runs in a dedicated `.venv-vilain-tamp`, enforced at runtime, so its
  model stack cannot alter the proposed method's environment.
- Fast Downward and VAL are built under `third_party/` and located relative to
  the repository root. The shipped configuration named absolute paths under
  one developer's home directory, which made the baseline unrunnable
  elsewhere; `_optional_path` now expands `~`/`${VAR}` and anchors relative
  paths at the repository root, while `ExternalToolPaths` still refuses a
  genuinely relative path so no planner resolves off a working directory.

### Observed failure mode

Across the first probe episodes (W1, W2, L1, K1) every run ended `EXHAUSTED`
with zero actions executed, and the mechanism is specific and reportable: the
model's generated `:init` omits a **static** compatibility predicate that no
action can establish. In Workshop that is `(fits ?fastener ?target)`, declared
in the domain and required by both `insert` and `drive` but absent from every
action's effects, so it can only come from the initial state. Fast Downward
correctly reports the initial state as a dead end (`Expanded 0 state(s)`,
`Dead ends: 1 state(s)`).

The baseline's own Corrective Planning diagnosed this correctly --
`RELAXED_GOAL_UNREACHABLE`, "all grounded achievers have unreachable positive
preconditions", naming `drive` as the potential achiever -- and the corrective
fact universe offered the missing predicate. The model still failed to select
it across three corrective iterations.

**A uniform zero is a weak result even when it is honest.** Before publishing
one, confirm that the initial-state prompt exposes the domain's static
predicates: "the baseline was never told a predicate it could not infer" and
"the baseline cannot do this task" are different claims, and only the second
is about the method.

## Physical execution controls

### Workshop: what "physical execution" covers

Workshop runs the **assisted grasp path**, for both the ground-truth oracle and
every baseline.  The robot navigates, reaches and actuates; the grasp itself is
completed by an equality constraint that attaches the payload without requiring
confirmed finger contact, and fastener insertion uses a compliant alignment
fixture.

**Do not describe Workshop execution as autonomous or contact-gated
manipulation.**  The accurate claim is robot-actuated execution with
constraint-assisted grasping and insertion.  Kitchen and Living Room are
contact-gated; Workshop is not, and the difference should be stated rather than
averaged away.

Artifacts say so: `execution_profile` is
`ASSISTED_GRASP_ROBOT_ACTUATED_GT_EXECUTION`, `INSERT_FASTENER` carries
`insertion_alignment_fixture_used: true`, and each assisted handle grasp
records `status: "LEGACY_ASSISTED_HANDLE_GRASP"` with
`bilateral_handle_contact_confirmed: false`.

#### Why contact-gated execution is not used here

This is measured, not assumed.  Contact-gated mode was enabled for the full
10-variant suite and fails:

| Variant | Failure |
|---|---|
| F0, F2, F4 | `workshop_long_phillips_driver` -- lift clearance, then no sustained bilateral finger contact (best streak 0; the gripper closes 8.2 cm short of the handle and the tool never moves) |
| F6, F7 | screw from `RIGHT_DRAWER` -- preclose pose off by 0.0781 m against a 0.075 m limit |
| F3 | `SCREW` -- gripper `ACTUATOR_STALL` |
| F1, F5 | all actions complete, then `driver_on_assigned_surface` and `hand_empty` fail |
| I0, I1 | correctly rejected (infeasible variants pass) |

The cause is that the Workshop grasp geometry was calibrated against the
assisted path, where the attachment constraint absorbs the positioning error.
Gating on real contact exposes that error per object and per base stance, and
the two are coupled: removing the strict-only 0.07 m drawer base offset brings
the screw's preclose inside its limit but then the fingers make no contact at
all, while keeping the offset makes contact but misses preclose by 0.6 mm.
Re-enabling contact gating therefore requires re-calibrating each grasp against
its stance, not a threshold change.

Kitchen runs contact-gated (`--strict-robot-execution`) and is unaffected.

#### Fixes retained from that investigation

These were found while enabling contact gating and are kept because they are
correct in either mode:

- The furniture-penetration audit exempts *finger-pad* contact with the single
  panel whose handle is being gripped.  Closing the gripper on a handle
  necessarily overlaps the panel it is mounted on (measured at 8.2 mm), so
  counting the commanded grasp as forbidden penetration was a false positive.
  Any other robot link touching that panel, and the pads touching anything else
  monitored, still fail.
- Contact-gated mode searches the same five IK seeds as the assisted path.
  Seeds are search breadth, not a physical criterion: every candidate passes
  the same collision check, so one seed did not make execution stricter, only
  more likely to fail.
- The caller's explicit `allowed_body_names` is honoured in both modes.  It
  previously discarded every named body without a free joint, which threw away
  the container the primitive had just named; entering an open drawer puts the
  finger tips 3-6 mm from its own front and floor, so reaching into any
  container was impossible.  The list is built per call and names only the
  container being manipulated, and the independent furniture audit still runs.
- The drawer lift aims at the measured apron plane instead of a fixed 0.18 m,
  which left the long driver's lowest point 3.1 cm *below* the apron -- still
  inside the drawer envelope.  Measured clearance for the screw improved from
  -13.3 cm to +4.4 cm.

The unaided insertion path is retained as `_strict_insert_fastener`, unused, so
the gap it measures stays stated: it leaves the screw 27 degrees off vertical,
7.8 mm lateral, tip 2.8 mm above the hole entry, against a gate of 0.05 rad,
3 mm and 8-18 mm of depth.

### Living Room


Reported physical success depends on the shared Living Room controller and
verifier, so these are part of the result definition and not incidental tuning:

- A manipulation stance must stay collision-free across the base controller's
  own settle tolerance, not only at the commanded point. A stance validated at
  a single point can settle a centimetre into a support shell, after which
  RRT* refuses to plan from that start and the robot is stranded for the rest
  of the episode.
- `LivingRoomMobileExecutor.plan` retreats to the nearest free pose when the
  base has already settled in collision, rather than failing every later
  action.
- A placed payload is judged at rest by pose drift over a window, not by an
  instantaneous velocity sample. Resting mesh payloads chatter in the contact
  solver: a saucer alternated between 0.06 and 0.48 rad/s on consecutive steps
  while its pose moved 0.03 mm per 500 steps, so a velocity threshold rejected
  placements that were provably static. The drift bounds are tighter than the
  velocity thresholds they replace.
- The gripper's closing twist rotates a payload in the grasp; a cup measured 0
  to 10.8 degrees. That offset is recorded at pick time and cancelled in the
  place command, so a commanded placement yaw is one the controller can
  actually achieve. The 12-degree placement yaw tolerance is then enforced for
  every payload and is not relaxed for round ones.
- **Self-collision allowances cover five base-link pairs, not two.** The
  Google robot permits bounded interpenetration between `base_link` and
  `link_shoulder` (-0.100), `link_bicep` (-0.050), `link_forearm`,
  `link_wrist` and `link_gripper` (-0.030 each), in
  `mujoco_scenes/generic_manipulation.py`. The last three were added by the
  phase-4 port and the widening is **intended**: it was part of that port's
  execution fixes, confirmed by the researcher on 2026-09-05. It is recorded
  here because it relaxes a physical validity criterion on the robot used in
  every Living Room episode, so it is part of the definition of a reported
  physical success rather than incidental tuning. All results in
  `runs/living_room/execution/feasible_i9_20260904*` were produced under the
  five-pair set. The mechanical justification for the three added pairs is not
  documented the way the original two are, and a stricter set would need the
  grid re-run to remain comparable.
- `benchmark_execution_result.json` records `mujoco_version`. Physical outcomes
  depend on the contact solver, so episodes from different engine builds are
  different result classes; `summarize_execution_batch` refuses to pool them.
  The repository pins `mujoco==3.3.6`.
- Every method reports a satisfied goal as terminal_status `GOAL_COMPLETE`, so
  failure-mode breakdowns keyed on that column stay comparable.
- A generation cut off by the token ceiling is reported as
  `MODEL_OUTPUT_TRUNCATED` and does not consume the episode's model-call
  budget. No plan was produced, so charging the budget would score the
  harness's ceiling as the method's planning failure: measured on a Living
  Room replan prompt, thinking-mode generation ran to the full 24576 tokens
  nine times in a row without closing its JSON, while every call that did
  complete needed under 7400. Truncation draws on its own bounded retry budget,
  in the same way a transport fault already did. These episodes are recorded,
  not dropped -- runaway reasoning on replan prompts is a real property of the
  checkpoint -- but they are a distinct failure mode and must not be pooled
  into a method's planning-failure count.

## Fair comparison controls

For every reported method and seed, freeze:

- the exact Phase-1 scene/evidence directory and full manipulable registry;
- initial MuJoCo state, five camera poses/resolution, goal, and action budget;
- persistent object/region IDs and the oracle instance-tracking procedure;
- physical skill implementation, IK/collision thresholds, and goal verifier;
- model name/revision, server version, prompt version, decoding mode, token
  limit, and random seed;
- maximum model calls and wall-clock planning limits.

The baseline runtime resolves all 15 manipulable objects in the frozen sample
registry, not just the 11 objects selected by the proposed functional planner.
Non-manipulable markers and ungrounded distractors are excluded consistently
because no method can send them to the physical skill backend.

Report at least task success, partial goal completion, VLM calls/tokens,
subgoal or plan attempts, TAMP time, motion-planning time, executed actions,
inspection count/order, failure codes, and wall-clock runtime. Report `paper`
and `model-native` decoding as separate conditions, never pooled; see
"Decoding conditions" above for the exact settings each one fixes.

## Required pre-run checks

```bash
cd ~/Documents/RRC/LH_Extension/V1
bash vlm_tamp_baseline/setup_pddlstream.sh

# `env -u PYTHONPATH` matters when ROS is sourced: its pytest plugin on
# /opt/ros hijacks collection and the run exits 0 having tested nothing.
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  baseline_common/tests vlm_tamp_baseline/tests owl_tamp_baseline/tests \
  retrieval_baseline/tests llm3_baseline/tests \
  mujoco_scenes/tests/test_kitchen_phase_b_execution.py \
  mujoco_scenes/tests/test_kitchen_phase_c_execution.py \
  mujoco_scenes/tests/test_kitchen_execution_entities.py \
  mujoco_scenes/tests/test_physical_dispatcher.py -q
```

Confirm the served checkpoint and the decoding actually sent, rather than the
configured defaults, before trusting a grid:

```bash
curl -s http://127.0.0.1:18000/v1/models | python3 -m json.tool
env -u PYTHONPATH .venv/bin/python -c "
from owl_tamp_baseline.planner import registry_sampling
print('thinking sampling:', registry_sampling('qwen35-9b', True))"
```

Each Living Room episode records what it sent: `method_manifest.json` for
VLM-TAMP and `model_trace.json` for OWL-TAMP. Read those, not the CLI
defaults, when reporting a condition.

Retain each run directory unchanged. `shared_observation_contract.json`, each
observation's `annotations.json`, raw and annotated frames, and each model
call's `model_visible_input` establish the input audit trail. Files below
`_private_evaluation/` contain labels used only by execution/evaluation and
must never be treated as model inputs. Parsed outputs, PDDLStream logs, action
telemetry, and the final evaluator result complete the paper audit trail.

For Living Room planning-only runs, additionally retain
`gt_sequence_comparison.json`. The expected GT file, semantic role map, and
MuJoCo backend resolution are evaluation/adapter-private. The VLM receives the
open-language goal, five ID-only annotated RGB views, semantic-neutral state,
history, and failure feedback—never the expected sequence or functional
assignment.

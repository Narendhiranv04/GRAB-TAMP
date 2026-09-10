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

## What each method sees

Two different exposures are involved and they must not be conflated.

**Images sent to the language model: three, for every method.** The grid runs
`--camera-counts 3`, and ViLaIn-TAMP is given the same count through its own
nested view subsets. This is the number that has to match, because it is the
evidence the model reasons over.

**Detector input for the proposed method: five views.** The proposed method
runs its detector over all five canonical cameras and grounds the resulting
candidates with geometric verifiers, while the language model still receives
only three images. That is a property of the method -- perception and
verification are its contribution -- and not an image-exposure advantage at
the model interface. It must be described that way rather than reported as
"five cameras", which would wrongly imply the model saw five.

A baseline has no separate detector stage: VLM-TAMP and OWL-TAMP receive
annotated views with persistent instance and region IDs, and ViLaIn-TAMP
performs its own object estimation with a model call. So the comparison is
three model-visible images on both sides, with the proposed method
additionally holding a detector and verifier stack. Stating the detector's
five-view input is necessary for the comparison to be read correctly.

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
- **The Workshop W1--W10 experiment is physically executed** as of
  2026-09-06. It uses a separate symbolic PDDLStream domain for `INSPECT`,
  `PICK`, `PLACE`, `INSERT`, and `FASTEN`, and every operator that domain
  emits is driven through the calibrated Google-robot skills by
  `mujoco_scenes/baseline_workshop_runtime.py`; the grid passes `--execute`.
  Physics decides each action and the symbolic rollout mirrors it, so the
  observation the model sees next reflects what actually happened
  (`MirroredWorkshopExecutor`). Storage contents remain hidden from the VLM
  until a physically successful inspect. Reported Workshop success is an
  execution-success claim.
  (Before 2026-09-06 this condition was planning-only and this clause said so.
  Workshop runs on the assisted-grasp path rather than contact-gated
  execution -- see "Why contact-gated execution is not used here" -- which is
  the one remaining difference from Kitchen and Living Room and must be stated
  when reporting.)

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
| Workshop | `UNIQUE_SEMANTIC_ALIAS_ONLY` | `UNIQUE_SEMANTIC_ALIAS_ONLY` |

Kitchen withholds region names because its regions *are* the search problem:
`D1`, `D2`, `B1`, `C1` and `C2` are closed storage whose contents are hidden
until inspected, and those exact tokens are on the forbidden list in
`mujoco_scenes/functional_tamp_pipeline/audit.py`. Naming them would tell the
model where to look.

Living Room has no closed or hidden regions, and its goal already names the
roles ("each person's fixed individual side table", "the fixed shared coffee
table"), so region aliases reveal nothing the instruction does not already
state. The `object_annotation` mode is identical in both.

Workshop names both, and its object labels are the most revealing of the three:
`OBJECT_LABELS` in `vlm_tamp_baseline/workshop_runtime.py` supplies "screw",
"manual screwdriver", "power screwdriver", "wooden hammer" and "frame joint",
and `REGION_LABELS` supplies "left drawer", "right drawer", "tool cabinet" and
"main workbench". These reach the model twice -- printed on the annotated
bounding boxes and as the alias field of each `Entity`/`Region` in the
textualized state.

**This matters for the Workshop comparison specifically.** The de-biased task
instruction says only "the compatible components required to complete the
fastening", but the annotation layer then hands the model the words "screw" and
"screwdriver" outright, so the category-noun leak the instruction fix removed
returns through the observation. It is faithful to VLM-TAMP and OWL-TAMP, whose
published inputs are annotated with object names, and it applies equally to
both -- but it is a real advantage over the proposed method's detector, which
grounds function from pixels, and over the retrieval control, which reads the
*raw* frames. Do not present Workshop grounding as solved from appearance by
the model-driven baselines: they are told what the objects are called, and only
the *compatibility* relation is left for them to infer.

Record the scene when reporting, because "the model was given semantic aliases"
means something different in each.

The old direct subgoal-to-action templates are not the baseline. They are
available only as `--refiner catalog-ablation`.

## OWL-TAMP correspondence

Preserved algorithmically:

1. Relaxed grounding enumerates reachable discrete actions while deferring
   continuous parameters optimistically.
2. A single multi-image VLM query produces a partial discrete sketch and goal
   facts. The image count is the grid's `--camera-count`, not a fixed five:
   the planner forwards whatever views the runner renders, and each request
   records them in `model_trace.json` as `camera_ids`. The reported execution
   grid sent **three** (verified in the recorded traces), matching VLM-TAMP.
3. `Executed(i)` preconditions/effects constrain the symbolic solution to
   contain that sketch as an ordered subsequence.
4. Separate VLM calls translate each sketched action's physical requirement
   into a restricted geometric constraint expression.
5. Refinement uses discrete search followed by bounded continuous sampling:
   500 samples per action (`PAPER_MAX_SAMPLES_PER_ACTION`) and a five-skeleton
   budget (`PAPER_MAX_SKELETONS`). **Exactly one skeleton is ever explored** in
   these domains: the discrete operators are deterministic, so the
   `Executed(i)`-constrained breadth-first search returns a single shortest
   skeleton and the budget is never reached. The budget is retained for
   provenance, and `skeletons_explored` is reported as 1 accordingly. Do not
   describe the search as exploring up to five skeletons here.
6. The paper's simulation condition is single-shot; it does not inherit
   VLM-TAMP's reprompt loop. The paper's real-robot appendix separately
   describes a receding-horizon observe--plan--execute policy.

Domain adaptations and reporting restrictions:

- The paper's simulation domains are replaced by the shared MuJoCo Kitchen and
  Living Room schemas. The model receives the same annotated views and
  alias-carrying textualized state as VLM-TAMP -- identical exposure, verified
  over the 2026-09-05 grid -- which is what makes the two columns comparable.
- **All three scenes are physically executed** -- Living Room and Kitchen via
  `--physical-execution`, Workshop via `--execute` -- through the same shared
  runtimes and the same goal verifiers as every other method, so reported
  success is an execution-success claim. This is a deliberate
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

In the K1--K12 trials, both visual baselines construct the variant directly
and use MuJoCo instance segmentation solely to assign persistent anonymous IDs
and semantic aliases in the configured views. (These trials were planning-only
before the physical-execution port; they are now executed, and the sentence no
longer restricts itself to a planning-only condition.) They do not consume the proposed framework's
Phase-1 object registry or functional witness. GT/backend ID translation occurs
only after planning. Report raw execution-vocabulary agreement and the shared
task-vocabulary normalization separately.

## Retrieval baseline correspondence

This is not a port of a published TAMP system. It is a deliberately minimal
open-vocabulary grounding baseline that isolates how far similarity-based
retrieval alone gets on the benchmark, with no language model in the loop.

1. The task structure is a fixed role template rather than a planned
   decomposition, one template per scene, in `retrieval_baseline/roles.py`:
   - **Living Room** (`LIVING_ROOM_ROLES`, 8 fillers): two personal supports,
     one shared support, two drink vessels, two under-dishes, one handheld
     control.
   - **Kitchen** (`KITCHEN_ROLES`, 9 fillers): one water source, one grounds
     source, one stirring implement, two drink vessels, two food vessels, two
     eating utensils -- driven through the 24-action ordered
     `KITCHEN_TASK_SKELETON`, because Kitchen is a sequence with one-gripper
     ordering constraints rather than a set of placements.
   - **Workshop** (`WORKSHOP_ROLES`, 2 fillers): one turning tool, one
     threaded part.
2. Each role is filled by CLIP ViT-B/32 image-text similarity between the
   role's function phrase and a crop taken from the raw, unannotated frames.
3. Role phrases describe function rather than naming the target's category.
   Audited phrase by phrase against the scenes' own semantic labels: no phrase
   contains the category noun of the object it is meant to select. Two contain
   generic hypernyms and are worth stating exactly rather than claiming
   absolute purity -- Workshop's `turning_tool` says "a hand tool", which is
   equally true of the wooden-hammer distractor and so discriminates nothing;
   Living Room's `under_dish` says "a shallow flat dish", a hypernym of the
   target rather than its name ("saucer"). Neither hands over the answer, but
   do not claim the phrases are noun-free.
4. Candidate supports are restricted to the runtime's registered support
   regions. The staging area is where payloads start and is not a placement
   target; electing it would let a missing table masquerade as a usable one.
5. A role that cannot be filled from distinct candidates yields
   `NO_RETRIEVED_ROLE_FILLER` and an infeasible verdict.
6. `uses_language_model` is false and every request counter is structurally
   zero, so this baseline is unaffected by the decoding conditions below and
   its numbers do not move with the served checkpoint.

4b. Required items may sit inside closed storage, and similarity gives no
   basis for choosing where to look, so the Kitchen and Workshop runners
   inspect every storage region in a fixed order **to exhaustion** before
   scoring. This is what makes their infeasible verdicts admissible:
   `infeasibility_proven()` requires every region to have been inspected.
   A closed region simply yields no candidates.

Physical execution runs the retrieved assignment through the same shared
skills as the other methods in all three scenes, so a wrong grounding appears
as a physically executed wrong plan rather than a planning-only mismatch.
Workshop routes through `MirroredWorkshopExecutor` rather than the physical
executor directly: without the mirror the planning runtime never learns that a
drawer opened, and the baseline returns a confident `INFEASIBLE` on a feasible
variant. Do not "simplify" that call site. Because
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

## ViLaIn-TAMP transport faults corrected 2026-09-07 (evening)

Two harness faults were destroying roughly half of all ViLaIn episodes under
`--decoding model-native`, both of them conditions no other tabled method
faced. Measured before the fix: Workshop 7 episodes launched -> 3 artifacts
(3 `TRUNCATED_RESPONSE`); Living Room 10 launched -> 5 artifacts
(2 `APITimeoutError`). A failed episode wrote **no artifact at all**.

1. **Truncation ended the episode instead of drawing on a retry budget.**
   `VLLMQwenTransport.complete` raised on the first
   `finish_reason == "length"`. This section's own truncation policy above
   requires the opposite -- "Truncation draws on its own bounded retry budget,
   in the same way a transport fault already did", and those episodes are
   "recorded, not dropped". VLM-TAMP implements that as
   `max_truncation_retries=2`; ViLaIn's port had no retry.
   Now `_completion_with_truncation_retry` redraws up to
   `_TRUNCATION_RETRY_ATTEMPTS = 3` (one call plus two retries, matching
   VLM-TAMP's budget) and reports exhaustion as the distinct
   `MODEL_OUTPUT_TRUNCATED`, which must not be pooled into a planning-failure
   count. The request is re-sent **unchanged**: a truncated draw is not
   evidence the request was wrong, and altering the budget here would move this
   baseline off the table's token limit. Redrawing is meaningful rather than
   superstitious because completion length is bimodal under thinking-mode
   sampling -- the measurement recorded above found completing calls needed
   under 7400 tokens while runaway ones hit the ceiling.
   `last_truncated_attempts` is recorded so an artifact shows whether the
   accepted completion needed a redraw.

2. **A 2x tighter transport deadline than the methods it is tabled against.**
   `configs/qwen_only.yaml` set `model_seconds: 300` where VLM-TAMP's and
   OWL-TAMP's planners both default to `timeout_seconds = 600.0`, and ViLaIn's
   OpenAI client is built with `max_retries=0`, so one slow call ended the
   episode. ViLaIn also makes the *slowest* calls in the grid: ~107 s measured
   for a four-contact-sheet request with thinking enabled, against a server
   saturated at `--max-num-seqs 8` with a queue behind it. Now 600, equal to
   the other two. This changes nothing about what is asked of the model.

Guarded by three tests in
`mujoco_scenes/baselines/vilain_tamp/tests/test_live_fm.py`
(`test_a_truncated_generation_is_retried_rather_than_failing_the_episode`,
`test_b_truncation_budget_is_bounded_and_reports_the_distinct_mode`,
`test_c_vilain_transport_deadline_matches_the_other_baselines`). ViLaIn's own
suite still passes: 363 passed, 3 skipped.

**Not touched, because it is a genuine ViLaIn failure:**
`InterpreterOutputError: camera_id must be a non-empty string` -- the model
returned a malformed detection. That is the method's own output, and it belongs
in its results.

**The 8 artifacts written before this fix remain valid.** Neither fault can
produce a *completed* episode -- both raise and abort -- so any artifact that
exists came from calls that neither timed out nor truncated. `--resume` keeps
them.

## Open: Workshop executor crashes on PLACE into a drawer

`workshop_ground_truth_execution._destination_position` raises
`ValueError: No assisted destination pose for LEFT_DRAWER` when a method plans
`PLACE(<tool>, left drawer)` -- returning a tool to a drawer. The exception
propagates out of the executive and kills the episode with no artifact, so the
harness's missing pose is recorded as nothing rather than as a failed PLACE.
Observed once in 200 Workshop episodes (`vlm_tamp` W8 seed 8, which is why
that grid stands at 199/200). Same class as the truncation fault above --
"recorded, not dropped" -- but it sits in the physical executor on the
assisted-grasp path, so it is left for a deliberate decision rather than
changed mid-grid.

## Blocking: Kitchen POUR/STIR can never succeed, so Kitchen feasible-success is not a capability result

**Do not report Kitchen `feasible_success_percent` as a measurement of any
method.**  Kitchen's goal is a seven-condition conjunction that includes two
pour relations and one stir relation, and in the baseline runtime those three
conditions are unsatisfiable by construction, for every method, on every
variant.  The reported `0/60` for VLM-TAMP, OWL-TAMP and Retrieval is a harness
artifact.

Two independent blockers are stacked, both unconditional:

1. **The POUR/STIR whitelist is empty.**  `KitchenPhaseCExecutionDispatcher`
   builds `expected_pairs` from its `frozen_plan` argument, and
   `baseline_kitchen_runtime.from_variant` constructs the dispatcher with
   `frozen_plan=[]` (`baseline_kitchen_runtime.py:456`).  So `expected_pairs`
   is `{'POUR': {}, 'STIR': {}}` and every POUR/STIR returns
   `POUR_TARGET_RESOLUTION_FAILED` / `STIR_TARGET_RESOLUTION_FAILED` before any
   geometry is consulted, whatever arguments the method chose.
2. **The registry carries no opening geometry.**  The same factory builds each
   registry row as `{"generic_object_id": ...}` and nothing else
   (`baseline_kitchen_runtime.py:656`).  `derive_target_opening` requires
   `opening_width_m`, `opening_length_m` and `cavity_depth_m` under
   `geometric_properties`, so even with the whitelist bypassed it raises
   `POUR_OPENING_GEOMETRY_UNAVAILABLE`.

Measured on K1 (`BaselineKitchenRuntime.from_variant`), all four functionally
valid pours and both valid stirs fail, and the geometry check fails
independently:

```
kettle -> mug            success=False status=POUR_TARGET_RESOLUTION_FAILED
kettle -> cup            success=False status=POUR_TARGET_RESOLUTION_FAILED
coffee_source -> mug     success=False status=POUR_TARGET_RESOLUTION_FAILED
coffee_source -> cup     success=False status=POUR_TARGET_RESOLUTION_FAILED
spoon_1 in mug           success=False status=STIR_TARGET_RESOLUTION_FAILED
spoon_2 in cup           success=False status=STIR_TARGET_RESOLUTION_FAILED
registry objects WITH geometric_properties: 0   (of 9)
derive_target_opening(object_0002) -> POUR_OPENING_GEOMETRY_UNAVAILABLE
```

Consistent with the 252 completed Kitchen episodes: POUR was attempted 368
times and succeeded 0 times, STIR was never reached, and the only effects ever
observed were `holding` and `placed`.

This is **not** a GT-grounding-mismatch issue.  The whitelist would gate POUR on
the ground-truth plan if one were supplied, but none is: the constraint binds
identically for a method whose grounding matches GT exactly.  The ground-truth
runner does not exercise this path either -- it passes `dummy_registry, []` and
replaces the ledger with `OraclePhaseCLedger`
(`kitchen_ground_truth_execution.py:389`) -- so Kitchen pour/stir physics has
never been wired to real geometry by any caller.

### What remains valid

* Kitchen `infeasible_rejection_percent` is unaffected: rejecting an infeasible
  variant requires no POUR.  VLM-TAMP 16/59, OWL-TAMP 0/60, Retrieval 6/6 stand.
* Kitchen `outcome_correct_percent` is dominated by the infeasible half; its
  feasible half is structurally 0 and must be described that way, not as a
  capability gap.
* **Workshop and Living Room are unaffected.**  Neither has a `frozen_plan`
  whitelist (`expected_pairs` does not exist in
  `workshop_ground_truth_execution.py` or `living_room_discovery_runtime.py`)
  and neither goal requires pour or stir.  Their 400 completed episodes stand.

### A fix is available and does not require ground truth

`exact_scene_geometry.extract_exact_object_geometry` already derives
`opening_width_m`, `opening_length_m` and `cavity_depth_m` from the live MuJoCo
mesh rim, which is perceptual rather than plan-derived -- it is the same routine
the feasibility oracle uses.  Populating `geometric_properties` from it clears
blocker 2 with no leakage.  Blocker 1 needs a decision rather than a patch:
POUR/STIR admissibility must stop depending on plan membership and be decided by
geometry, so that the pair a method actually chose is what gets executed and
judged.  Until both are done, Kitchen's feasible column is not reportable.

## What the Workshop goal is, and what it is not (2026-09-09)

The published Workshop instruction has three clauses:

> Identify the compatible components required to complete the fastening at the
> marked workbench location, complete the fastening, **and leave any reusable
> equipment used for the task safely on the workbench.**

Only the middle clause is visible in the physical joint state.
`WorkshopPhysicalExecutor.goal_satisfied` is `repaired_joint == TARGET_JOINT`
and nothing else, so it is not the goal; it is the fastening clause of it.

VLM-TAMP handed that clause to the executive as its goal verifier. Because the
executive returns as soon as its verifier passes, the omission it rewarded was
also the omission it caused: of 400 Workshop episodes, 8 ever fastened, and all
8 ended with the driver still in the gripper, stopped one action short of the
goal they were credited with. The environment's own predicate had recorded
`goal_satisfied: false` beside `terminal_status: GOAL_COMPLETE` in every one.

OWL-TAMP and the Retrieval baseline drove their control loops from the full
predicate but reported the verdict from the joint alone. Neither reported
number moves -- neither ever fastened -- but a run that pursues one definition
and reports another is a defect regardless of whether the arithmetic changes.

**What may be claimed.** Workshop success is `workshop_goal_reached`: the
physical scene is authoritative on the fastening, and the runtime predicate
carries the clauses the joint cannot see (empty gripper, driver returned to the
work surface). A number scored on the joint alone may not be reported as
Workshop success. The 100 VLM-TAMP episodes recorded under the old definition
are retired to `runs/_superseded/`.

**Goal coverage** is the three terminal conditions in that instruction:
fastener seated, joint fastened, driver back on the workbench with an empty
gripper. Which driver did it is deliberately not checked against ground truth,
for the reason the Living Room established -- demanding GT's specific
assignment marked 30 of 95 correct solves wrong -- and because the physical
FASTEN skill already rejects an incompatible driver, so a recorded fastening is
a compatible one.

## Limitation: ROBUST-TAMP and the 24576-token ceiling (2026-09-09)

`--max-tokens` is 24576 for every model-driven method, and it stays there. This
is recorded as a limitation of the ROBUST-TAMP framework rather than corrected,
because raising the ceiling for one method would make the comparison measure
the ceiling instead of the method.

ROBUST-TAMP pays for it more than the others. Its planner must emit strict JSON
containing exactly `status` and `actions`, with explanations and markdown
forbidden by its system prompt, after a thinking pass. Measured on the retired
Kitchen leg: 48 of 519 planner calls reached the ceiling before emitting any
JSON at all, one of them after 450 s, and the reply is then unusable. A further
403 of 1570 calls across the retired Kitchen and Workshop legs were rejected
for carrying top-level keys other than those two.

The VLM-TAMP and OWL-TAMP legs do not show this: across 620 episodes of
Kitchen, Living Room and Workshop grids their truncation count is 0. The
ceiling binds on ROBUST-TAMP specifically, because of what its output contract
demands after a long generation.

**What may be claimed.** ROBUST-TAMP's scores are reported at the same decoding
condition as every other method, and its output-format brittleness under that
condition is part of the result. It may not be reported as though it ran under
a different ceiling, and the other methods' numbers may not be re-run at a
higher one for comparison.

## Measured: where ROBUST-TAMP's decode actually goes (2026-09-09)

Alongside the 24576-ceiling limitation above. Measured from `latency_s` in
`model_calls/*.json` across the two ROBUST-TAMP re-run legs, over 48.7
GPU-hours of recorded decode:

| cause | calls | hours | % of all decode |
|---|---|---|---|
| transport: timeout/unreachable | 88 | 14.7 | 30.1% |
| schema: wrong top-level keys | 182 | 11.5 | 23.5% |
| plan validation: gripper occupied | 195 | 11.0 | 22.5% |
| plan validation: not visible | 44 | 3.7 | 7.5% |
| plan validation: not holding | 51 | 2.7 | 5.5% |
| truncated at the 24576 ceiling | 1 | 0.1 | 0.3% |
| **total producing nothing usable** | **565** | **43.7** | **89.8%** |

**Read it in two parts; they have different owners.**

About 59% is the method's own invalid output -- schema non-compliance plus
plans rejected by precondition validation. This is a property of ROBUST-TAMP
under the fixed decoding condition and may be reported as such. The schema
share is near-misses rather than nonsense: of 54 rejections inspected, 19
omitted `status`, 13 used `plan` where the contract says `actions`, 5 sent
`plan, status`, 5 echoed `output_schema`, 2 sent a bare unwrapped action, and 1
used `PLAN`. The model gets the structure right and the key names wrong.

About 30% is request timeouts, which scale with concurrency and are therefore
partly an artifact of how the grid was scheduled: Workshop ROBUST-TAMP reached
1.45 timeouts/episode at 8 workers among 23 concurrent episodes, 1.00 at 4
workers, and 0.53 at 10 workers with the GPU to itself -- the last matching the
retired run's 0.54. **This share may not be reported as a property of the
method.** Report the figure measured with the leg running alone.

Truncation at the ceiling is 0.3% of decode, against 0.53 per episode in the
retired Kitchen leg. So the token ceiling -- the thing originally suspected of
holding ROBUST-TAMP back -- is not where it loses its compute. Output
discipline is. That is the substantive reason `--max-tokens` stays at 24576 for
every method, beyond the fairness argument.

### Instrumentation gap: this comparison cannot yet be made across methods

VLM-TAMP and OWL-TAMP record no usable `latency_s` (0 of 395 calls in the
Workshop VLM-TAMP leg), and no method records token usage at all. So wasted
decode is measurable only *within* ROBUST-TAMP. An earlier working table
showed VLM-TAMP at "0.0% wasted"; that was missing data, not a finding, and
must not be reported. To make this a real efficiency column beside "Raw VLM
requests", add `latency_s` and the response's token usage to every method's
model-call artifact.

## Workshop ROBUST-TAMP: the ceiling, not the contention (2026-09-09, final)

The section above said the ~30% timeout share of ROBUST-TAMP's wasted decode
scaled with concurrency and must be re-measured with the leg running alone.
That re-measurement is done, and it revises the conclusion rather than
confirming it.

Running alone at 10 workers, request timeouts fell to **0.00/episode** -- so
that share was indeed contention. But truncation rose in their place, because
requests that previously died on the clock now run to completion and reach the
token ceiling instead. Over the full 100-episode leg, 707 planner calls:

| outcome | calls | share |
|---|---|---|
| plan validation rejected the plan | 208 | 29% |
| schema: wrong top-level keys | 158 | 22% |
| truncated at the 24576 ceiling | 127 | 18% |
| timeout / unreachable | 54 | 8% |
| usable | 160 | 23% |

84.8% of its decode produces nothing usable. Truncation concentrates sharply on
the infeasible variants -- **3.15 per episode on W9--W10 against 0.94 on
W1--W8** -- which is consistent with their mechanism: an infeasible task has no
goal state to terminate on, so the planner reasons until something stops it.

**What may be claimed.** Workshop ROBUST-TAMP's 1.0% outcome correctness is
bounded by the decoding condition as much as by the method, and must be
reported with that stated. It is not evidence that the approach cannot reject
an infeasible Workshop task; it is evidence that under a 24576-token ceiling it
does not get far enough to try -- it recognised 1 of 20. The ceiling is held
uniform at 24576 for every method precisely so the comparison does not become a
measurement of the ceiling, and the honest way to report this row is to say so
rather than to raise it for one method.

The earlier instruction stands with its reason corrected: do not report the
timeout share as a property of the method, because it was contention. Do report
the truncation share, because it is not.

## Kitchen is bounded by the manipulation controller, not by the methods (2026-09-10)

An earlier note here, and the caption of `results_table_split.tex`, said the
uniform $0.0\%$ Kitchen feasible success was "a failure to search rather than
a failure to plan". That claim is withdrawn. Search *is* required and *is*
never performed, but it is not what produces the zero.

Ground truth was re-run on the current tree to establish whether the plans are
executable at all (`runs/gt_recheck_20260910`, `runs/gt_workshop_recheck_20260910`):

| suite | result |
| --- | --- |
| Kitchen GT, 6 feasible variants | 5 SUCCESS; F2_HIDDEN_SOUP_BOWL fails `GRASP_FAILED` at 22/25 |
| Workshop GT, 8 feasible + 2 infeasible | 10/10; every feasible variant completes, both infeasible correctly confirmed |

This reproduces `runs/gt_kitchen_locked` (2026-09-07) exactly, so the outcome
is stable, not a lucky seed.

**Ground truth succeeding does not mean the executor will run any plan.** The
Kitchen GT runner records its profile as
`PHYSICAL_PRIMITIVES_WITH_ASSISTED_RECOVERY`, and that name is accurate. Three
things it does that a baseline does not:

1. **Pre-validated placement coordinates.** `_allocate_staging_spot` in
   `mujoco_scenes/kitchen_ground_truth_execution.py:583` selects a countertop
   XY from a per-role table whose comments record that the coordinates were
   found by physical trial ("Reuse the source-return locations physically
   proven by K1"). A baseline calls `phase_b.place(object_id, "countertop")`
   and gets whatever `placement_resolver.resolve` returns.
2. **The controller timeout is caught.** GT wraps `phase_b.place` in
   `try/except Exception` (`kitchen_ground_truth_execution.py:1763`) and
   downgrades a timeout to a failed action. The baseline path
   (`kitchen_phase_b_execution.py:280` -> `kitchen_object_manipulation.py:4478`)
   calls `_step_until_stable_mode()` unguarded, so `RuntimeError(
   "MANIPULATION_EXECUTION_TIMEOUT")` propagates to the dispatcher's generic
   handler and ends the episode as `internal_error`. `PICK` does wrap it
   (`kitchen_object_manipulation.py:4176`); `PLACE` does not. The asymmetry is
   the whole difference between a recoverable action failure and a dead
   episode.
3. **A verified release is accepted.** When the planned placement fails but the
   object left the gripper, GT runs `validate_stable_placement` and returns
   `RELEASED_PLACEMENT_VERIFIED`. The baseline has no such route.

Measured through the baseline path, over the 60 feasible VLM-TAMP Kitchen
episodes in `runs/kitchen/execution/pour_rerun_20260908`:

| skill | ok | failed |
| --- | --- | --- |
| PICK | 60 | 11 |
| POUR | 44 | 11 |
| PLACE | 3 | 52 |

54 of 60 episodes (90%) terminate on a physical manipulation failure: 43
`MANIPULATION_EXECUTION_TIMEOUT`, 10 `POUR_ALIGNMENT_FAILED`. `F0_ALL_VISIBLE`,
which requires no search whatsoever, fails identically. ROBUST-TAMP dies the
same way, on the same action shape: PICK ok, POUR ok, `PLACE(_, countertop)` ->
`MANIPULATION_EXECUTION_TIMEOUT`, three actions into a 19-action accepted plan.

`_step_until_stable_mode(maximum_steps=30000)` at a 0.002 s timestep is 60
seconds of simulated time, so this is not a tight budget being missed; the
executor never reaches `holding`, `idle` or `failed` at all.

Two further corrections to earlier claims:

- **STIR is proposed.** A previous session recorded that the baselines never
  emit STIR. They do: VLM-TAMP's K1 plan contains
  `STIRRED{target_id: object_0003, tool_id: object_0008}`, and six episodes
  terminate on the `STIRRED` subgoal. `stirred` is 0/120 because the action
  times out, not because it is never attempted.
- **`placed` cannot be scored from these runs even if PLACE worked.** 51 of 55
  baseline PLACE attempts target `countertop`; none targets `serving_area`,
  which is the only destination `baseline_kitchen_runtime.py:263` scores as a
  `placed` effect. The Kitchen `placed` denominator of 0/360 is therefore
  jointly bounded by the controller failure and by plan content.

What may be claimed: Kitchen feasible success and goal coverage are reported as
measured, and are a lower bound on method capability bounded by a harness
limitation. What may not be claimed: that Kitchen distinguishes the methods'
planning ability. Use the plan-level coverage in `scripts/planning_metrics.py`
for that, which scores the same goal conditions against the plan.

### Workshop is not the same story

Workshop separates the baselines at four different stages, and only one of them
is execution-bounded:

| method | where it stops |
| --- | --- |
| VLM-TAMP | reaches execution: 139 inspections, PICK 78 ok/89 fail, PLACE 36 ok/13 fail, FASTEN 10 ok |
| OWL-TAMP | emits a sketch containing only `INSPECT`; 216 inspections, zero manipulation actions ever proposed |
| ROBUST-TAMP | 72 of 100 episodes execute exactly one action, then `inference_failed` under the 24,576-token ceiling |
| ViLaIn-TAMP | `execution_status: NOT_RUN_NO_SELECTED_PLAN` in all 100; its 10 correct outcomes are all infeasible rejections |

Workshop GT passing 10/10 on the current tree confirms the scene's physics is
not the constraint there.

## Five harness faults fixed, and what they invalidate (2026-09-10)

A line-by-line audit of the 51,696 JSON artifacts in the canonical roots
(`scripts/audit/`) established that three of the four Kitchen zeros were ours,
not the methods'. What the audit cleared, and what it found:

**Clean.** Every file parses. The Table I goal string is byte-identical in all
1,292 episodes. All 2,371 requests went out at 24,576 tokens to `qwen35-9b`
with thinking enabled under one sampling configuration. No private artifact,
goal contract, ground-truth action file or internal variant name -- key or
value -- appears in anything the model was shown. `outcome_match` never
contradicts its two outcomes, no success is recorded without full coverage,
none on an infeasible variant, no duplicate trials, seeds 0-9 complete
everywhere except Retrieval.

**Fixed.**

| # | fault | what it invalidated |
| --- | --- | --- |
| 1 | OWL read an object's region from `region_id`/`location`; Kitchen publishes `source_region` | all 120 OWL Kitchen episodes |
| 2 | OWL's Executed(i) search was breadth-first over the grounded set | OWL Kitchen, and any long-sketch scene |
| 3 | ViLaIn's detection depth was the box median, which is background | every ViLaIn episode in all three scenes |
| 4 | Kitchen PLACE stranded the executor after a timeout | VLM and ROBUST Kitchen |
| 5 | Workshop ROBUST never wrote `history`; ROBUST discarded replies for key naming | ROBUST Workshop telemetry, and 43% of all ROBUST replies |

Fault 1 is the third instance of the same class: Kitchen speaks a different
observation dialect from the Living Room and the Workshop, and each consumer
that reads only one spelling fails silently. The earlier two were
`holding`/`held_object` in `baseline_observation_bridge.py` and the ViLaIn
stage-id mismatch. **The three scenes should be made to publish one
vocabulary**; until they are, every new consumer is a candidate for the same
bug.

Fault 2 is worth separating from fault 1 because it only became visible after
fault 1 was fixed: with an empty world model the search dead-ended after 32
expansions and looked like an unreachable goal; with locations restored it ran
out of budget at sketch progress 3 of 14. Replaying the recorded sketches,
31 of 56 feasible Kitchen sketches now yield a skeleton where 0 of 120 did.

Fault 5's second half is a fidelity decision rather than a defect, and is
recorded as one: the two-key reply contract is ours, the system prompt asks
for exactly `status` and `actions`, and 394 of 923 replan requests were
rejections for naming alone -- 111 replies put the action list under `plan`,
120 omitted `status` while carrying `actions`, 47 sent `plan` beside a valid
`status`. No plan was judged in any of them. Unambiguous aliases are now
accepted; every repair is counted on `DiscoveryPlanner.shape_repairs` so the
effect on the reported numbers stays measurable rather than becoming
invisible leniency.

**Known and not fixed.**

- Retrieval's Kitchen goal contract is the sentinel `__planning_goal_unset__`
  in every variant, it has one seed per variant and no Living Room or Workshop
  runs. It was never wired up and cannot be reported.
- 12 OWL Workshop episodes (W1 all seeds, W2 seeds 0-1) ran with
  `shared_observation_contract.physical_execution: false` while the result
  file says `true`; the flag flipped mid-leg. Behaviour is indistinguishable
  because OWL Workshop never manipulates, but the leg was not run under one
  configuration. Covered by the re-run.
- `predicted_outcome` takes a third value, `UNRESOLVED`, and it dominates the
  infeasible trials -- ROBUST 35/40, VLM 32/40, ViLaIn 17/40, OWL 5/40. It is
  neither a rejection nor a false completion, so those two columns correctly
  do not sum to 100. The caption must say so.
- One Kitchen PLACE timeout costs 796 s of wall clock (measured,
  `runs/placefix_smoke`). Kitchen's cost is dominated by this, not by
  inference.

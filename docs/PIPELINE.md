# GRAB-TAMP — stage-by-stage reference

Written so a later change can be aimed at the right stage instead of a
plausible-looking one. For the method overview, setup and reproduction
commands, see the [README](../README.md).

---

## 1. What a single trial does

One trial is one (domain, variant) pair taken from the task instruction to a
validated symbolic plan. Exactly **one foundation-model call** is made, at the
start; everything after it is deterministic given that response.

```
instruction + 3 rendered RGB views
        │
        ▼  ONE model call  (schema v3, temperature 0.6, 3 views)
raw response          fm_diagnostics/fm_call_001.json
        │
        ▼  structural sanitizer            structural_sanitization.json
        │  v3 → canonical document, dangling references repaired
        ▼  semantic compilation            functional_specification.json
        │  ── this is where free-form wording becomes fixed vocabulary ──
        │  cue tables first; zero-shot NLI fallback only where they return
        │  nothing, so a resolvable phrase takes the identical path
        │     roles        coffee_cup      → coffee_container
        │     capabilities "pour"          → TRANSFER_CONTENT_TO_CONTAINER
        │     relations    "placed_on"     → FITS_SET_ON
        │     regions      "table drawer"  → D1
        ▼  G_F  functional requirement graph
        │
        ▼  search contract frozen          run_manifest.json: region_order_used
        │  region inspection order fixed before any perception
        ▼  perception + region search      observed_search/, observed_scene_graph.json
        │  open-vocabulary semantics → point-cloud geometry → G_O
        ▼  joint grounding  φ: G_F → G_O   graph_grounding_result.json
        │
        ▼  ≤ 1 A* invocation               symbolic_problem.json
        ▼  independent symbolic validation result.json
```

Two properties are load-bearing and easy to lose in a refactor.

**The search contract is frozen before perception runs.** The inspection order
is committed at the point `search_contract.py` builds it, and no later stage may
revise it. A system allowed to reorder after seeing what is inside a container
is not solving the same problem, and its inspection-cost numbers are not
comparable.

**The model is never consulted twice.** There is no resampling, no repair call,
no fallback prompt. Any failure therefore stays a failure of the method, which
is what makes the reported figures attributable at all.

## 2. Where a trial can fail, in pipeline order

First-cause attribution assigns each failing trial to the **earliest** stage
that broke; that ordering is why the later stages look small. The taxonomy is
`functional_tamp_pipeline/outcome_classifier.py`, which calls itself
authoritative and is the only place the decision is made.

| stage | outcome category | what it means |
|---|---|---|
| Task Specification | `TASK_SPECIFICATION_FAILURE`, `FM_RESPONSE_FAILURE` | no usable document: unparseable, or arrived and wrong about the task |
| Graph Compilation | `GRAPH_COMPILATION_FAILURE` | the response could not become an executable graph — unresolved roles or semantics, disabled groups, unmapped required operations |
| Object Discovery | `OBJECT_DISCOVERY_FAILURE` | search exhausted without enough role-compatible observed individuals: the region was opened, the object was never found |
| Functional Assignment | `FUNCTIONAL_ASSIGNMENT_FAILURE` | the observed individuals cannot jointly satisfy the constraints: objects found, no consistent role assignment |
| Planning | `PLANNING_FAILURE` | A\* found no complete sequence over a complete grounding |

The discrimination between the middle two is one line in the classifier —
`search_exhausted and not individual_candidates_sufficient` — and it is worth
knowing, because an ad-hoc reimplementation of this taxonomy attributed the
workshop's assignment failures to discovery and inverted the two largest cells
of the table.

`scripts/make_paper_tables.py` reports this table and makes one documented
adjustment: a trial whose outcome is `SUCCESS` but which did not satisfy every
reference goal is attributed to Functional Assignment, since a plan was
produced and executed and the end state is still wrong.

**Graph compilation is the dominant blocker.** Not producing the specification,
and not planning over it — turning the model's own words into an executable
graph.

## 3. The four vocabularies

| layer | target vocabulary | size |
|---|---|---|
| role | `coffee_container`, `soup_eating_utensil`, `driver`, `fastener`, … | 6 / 6 / 3 per domain |
| capability | `STIR_COFFEE`, `TRANSFER_CONTENT_TO_CONTAINER`, `FASTEN_JOINT`, … | 7 total |
| relation | `INSERTABLE_IN`, `REACHES_BOTTOM`, `FITS_SET_ON`, … | 9 binary |
| region | `D1 D2 C1 C2 B1`, `LEFT_DRAWER RIGHT_DRAWER TOOL_CABINET` | 5 / 0 / 3 |

Region aliases must stay grounded in the scene geometry. An earlier alias table
carried entries that the scene contradicted — one wall cabinet described as a
base cabinet, and two drawers distinguished as "top" and "bottom" when
`assets/kitchen_base.xml` places both at z = 0.46 — together with side-less
aliases that sent every generic "wall cabinet" to one side, 107 against the
other's 39. The effect was not a modest bias: the FM arm ranked only 1.10 of 5
kitchen regions, the deterministic system completion supplied the rest in
canonical order, and an inspection-order ablation run on top of that was
measuring canonical order against random order while reporting it as
FM against random. After grounding the aliases in geometry the arm ranks 2.43
of 5 and the per-region counts come out symmetric.

## 4. Changing it without changing it

Every ablation runs as a **shadow**: a context manager patches named functions
in process, runs the unmodified evaluator, and restores them on exit. No
pipeline file is edited, and each arm replays the frozen responses.

| shadow | patches |
|---|---|
| `mujoco_scenes/fm_evidence_ablation.py`, `fm_ablation_shadow.py` | `grounding.ground_graph` |
| `mujoco_scenes/fm_search_order_shadow.py` | `run.search_until_satisfied`, the domain `run_to_plan` functions |
| `mujoco_scenes/fm_worst_case_order.py` | the inspection order |
| `mujoco_scenes/fm_zs_canonicalization_shadow.py` | the four canonicalization resolvers — **part of the reported method**, run with `--baseline` to turn it off |

Three rules keep a shadow honest.

- **Augment, never replace.** Consult the existing resolver first and act only
  where it returns nothing, so a trial the current system handles takes the
  identical path and the arm cannot regress what already works.
- **Report the counts.** A patch that silently fails to apply produces a second
  copy of the baseline and reads as a clean null result. Every shadow counts how
  often it fired and warns when it never did.
- **Verify the patch point on one trial before running the matrix.** Twice a
  plausible function turned out not to be the one that decides: `_map_role` is
  overridden by `resolve_role_type_hypotheses`, and `_participant_slot_fit` is
  never reached because `interpret_operation` rejects the operation earlier on
  endpoint signature. Each fired hundreds of times, changed nothing, cost hours,
  and was visible in a single trial.

Timing needs the same discipline. Grounding time is only interpretable within
one machine, and arms must run back to back with nothing else competing — one
attempt ran a replay pass concurrently, drove the machine into swap, and its
trials had to be discarded. Read the Living Room cells first: that domain
declares no inspectable regions, so every arm runs it identically, and any
difference it still shows is the measurement floor rather than an effect. On
one occasion that floor was 18.3%, enough to consume most of the difference
being claimed.

Prefer a cost measure with no wall clock in it where one exists. The reported
opening cost is measured per-region actuation time multiplied by which regions
were opened, so it cannot drift.

## 5. Ground-truth isolation

Ground truth is used only for offline scoring. `scripts/audit_no_gt_leakage.py`
fails if anything under `functional_tamp_pipeline/` reads ground-truth
feasibility, identities, roles, success, inventories or expected answers, or
branches on a variant name or trial index. Run it after any change to the
pipeline, not only before release.

## 6. Known open problems

**Role collisions.** `AMBIGUOUS_ROLE_MAPPING` (75 occurrences) is two distinct
model roles resolving to the same canonical role, and it is the largest single
sub-cause of the dominant failure stage. An augment-only fallback never fires
for it, because the mapper already returned an answer — the answer is the
problem. Addressing it means either enriching the canonical vocabulary so the
two roles separate, or making resolution reject a collision rather than commit
to one side.

**One model role, two canonical roles.** Living Room's characteristic failure.
The model writes a single `placement_surface` whose own stated function covers
both the refreshments and the entertainment control:

```
placement_surface  REGION  count=2
  "Supports the refreshment items and the entertainment control."
```

That role resolves to `PERSONAL_CUP_SAUCER_REGION`.
`SUPPORT_ENTERTAINMENT_CONTROL` then finds no capability whose target signature
accepts it, the operation is dropped as
`NO_CAPABILITY_SIGNATURE_ACCEPTS_THESE_PARTICIPANTS`, and the remote is left
with nowhere to go. The affected trials place the drinkware correctly and stop
two actions short.

The model output is complete and correct here; the loss is entirely in
canonicalization. Fixing it means role resolution becoming **operation
relative** rather than one canonical name per model role — a design change
rather than a patch. Note for anyone attempting it: the capability that needs
rescuing carries the region as its **source**, not its target
(`source = SHARED_REMOTE_REGION/REMOTE_SUPPORT`,
`target = REMOTE/REMOTE_CONTROL`), so a substitution written against the target
slot never fires. The safety argument that a region is only ever a destination
does not hold for this signature.

**`NO_GLOBAL_REGION_ASSIGNMENT`.** Some feasible Living Room trials are declared
infeasible with this message, which says what failed but not which constraint
could not be met, so it cannot be acted on without re-deriving the assignment
by hand.

# Evaluation-only ground truth

Everything in the `GT_*` directories is read **only** by the offline scoring
scripts. Nothing under `mujoco_scenes/functional_tamp_pipeline/` may read any
of it, and that is enforced rather than asserted:

```bash
PYTHONPATH=. python3 scripts/audit_no_gt_leakage.py
```

The audit fails if the pipeline reads ground-truth feasibility, object
identities, roles, success, container inventories, action sequences or expected
answers, or branches on a variant name or a trial index.

| directory | contents | consumed by |
|---|---|---|
| `GT_GOAL_COVERAGE/` | per-variant goal definitions | goal coverage |
| `GT_VALID_ROLE_ASSIGNMENTS/` | admissible role→object bindings, identity signatures | functional assignment coverage |
| `GT_ROLE_ASSIGNMENTS/` | reference role assignments | role scoring |
| `GT_everything/` | reference action sequences and role assignments per variant | end-to-end action match |

Regenerate the two scored artifacts with:

```bash
PYTHONPATH=. python3 scripts/build_evaluation_gt.py
```

The reference action sequences are the basis of the end-to-end metric, which
compares the produced action multiset against the reference multiset. Matching
it is *sufficient but not necessary* for satisfying every goal: a trial can
reach the goal state by a different sequence, and a few do.

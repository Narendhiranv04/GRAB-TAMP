# Joint semantic–geometric grounding report

- Scene: `S1_joint_stir_counterexamples`
- Task: `stir_contents`
- Source run: `/home/naren/RA_iiith/runs/my_complete_ablation_report`
- Same saved observation evidence used by all modes: `true`
- All expected outcomes matched: `True`

![Ablation comparison](ablation_comparison.png)

## What was done

The scene was captured once per stage with five region-facing cameras. RGB supplied semantic evidence; metric depth and instance masks supplied fresh stage-local point-cloud evidence. Geometry-only and semantic-only are diagnostic acceptance ablations over those same saved observations. Only joint mode is the production decision.

## Outcomes

| Mode | Completion | Selection | Correct? |
|---|---:|---|---|
| Geometry-only diagnostic | stage 0 | `{'mixing_container': 'bowl', 'mixing_tool': 'pen'}` | False |
| Semantic-only diagnostic | stage 1 | `{'mixing_container': 'bowl', 'mixing_tool': 'spoon'}` | False |
| Joint semantic + geometric grounding | stage 2 | `{'mixing_container': 'bowl', 'mixing_tool': 'fork'}` | True |

## Visualizations

### Geometry-only diagnostic

Ignores semantic compatibility. It therefore accepts the marker/pen at INITIAL because its shape is elongated, insertable, and long enough. This is intentionally an incorrect diagnostic result.

![geometry-only](ablations/geometry_only/geometry_only.gif)

### Semantic-only diagnostic

Ignores unary and pairwise geometry. It therefore accepts the rank-1 oversized spoon after D1 even though its measured cross-section does not fit the bowl opening. This is intentionally incorrect.

![semantic-only](ablations/semantic_only/semantic_only.gif)

### Joint semantic + geometric grounding

Requires semantic compatibility, unary geometry, pairwise geometry, and distinct role assignments. It rejects the marker semantically, rejects the oversized spoon geometrically, and selects the fork at D2.

![joint](ablations/joint/joint.gif)

## Machine-readable evidence

- `offline_ablation_evaluation.json`
- `report_data.json`
- Original run: `/home/naren/RA_iiith/runs/my_complete_ablation_report`

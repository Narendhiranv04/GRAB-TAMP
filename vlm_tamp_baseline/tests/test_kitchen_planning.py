from __future__ import annotations

from types import SimpleNamespace

from mujoco_scenes.baseline_kitchen_runtime import (
    KitchenEffectLedger,
    KitchenGoalContract,
)

from vlm_tamp_baseline.kitchen_planning_runtime import (
    KitchenPlanningState,
    canonical_kitchen_actions,
    compare_kitchen_actions,
    normalize_kitchen_actions,
)
from vlm_tamp_baseline.planner import VLMTAMPPlannerConfig
from vlm_tamp_baseline.run_kitchen import _method_manifest, build_parser


def test_kitchen_planning_cli_exposes_private_gt_condition() -> None:
    arguments = build_parser().parse_args(
        [
            "--goal", "goal", "--output-dir", "run",
            "--planning-only", "--variant", "K1",
        ]
    )
    assert arguments.planning_only
    assert arguments.variant == "K1"
    assert arguments.max_model_calls is None
    assert arguments.seed == 0


def test_method_manifest_is_available_before_planning_result() -> None:
    arguments = build_parser().parse_args(
        [
            "--goal", "goal", "--output-dir", "run", "--planning-only",
            "--variant", "K1", "--camera-count", "3",
        ]
    )
    manifest = _method_manifest(
        arguments,
        VLMTAMPPlannerConfig(
            base_url="http://localhost/v1",
            model="test-model",
            enable_thinking=False,
        ),
        max_model_calls=1,
        inventory_object_count=9,
    )
    assert manifest["planning_rounds"] == 1
    assert manifest["raw_vlm_requests_per_round"] == 2
    assert manifest["camera_count"] == 3


def test_canonical_kitchen_actions_translates_ids_after_planning() -> None:
    history = (
        {
            "action": {"skill": "INSPECT", "arguments": {"region_id": "C1"}},
            "success": True,
        },
        {
            "action": {"skill": "PICK", "arguments": {"object_id": "object_1"}},
            "success": True,
        },
        {
            "action": {
                "skill": "PLACE",
                "arguments": {"object_id": "object_1", "region_id": "countertop"},
            },
            "success": True,
        },
    )
    assert canonical_kitchen_actions(history, {"object_1": "spoon_body"}) == [
        {"operator": "INSPECT", "arguments": ["C1"]},
        {"operator": "PICK", "arguments": ["spoon_body"]},
        {"operator": "PLACE", "arguments": ["spoon_body", "countertop"]},
    ]


def test_task_level_normalization_removes_execution_only_vocabulary() -> None:
    expected = [
        {"operator": "OPEN", "arguments": ["D1"]},
        {"operator": "CLOSE", "arguments": ["D1"]},
        {"operator": "PICK", "arguments": ["spoon"]},
        {"operator": "PLACE_SERVING_UTENSIL", "arguments": ["spoon", "bowl"]},
    ]
    assert normalize_kitchen_actions(expected) == [
        {"operator": "INSPECT", "arguments": ["D1"]},
        {"operator": "PICK", "arguments": ["spoon"]},
        {"operator": "PLACE", "arguments": ["spoon", "bowl"]},
    ]
    comparison = compare_kitchen_actions(
        normalize_kitchen_actions(expected), expected
    )
    assert comparison["shared_task_vocabulary"]["exact_sequence_match"]


def test_private_goal_verifier_checks_task_relations_not_gt_assignment() -> None:
    labels = {
        "cup": "cup", "mug": "mug", "bowl_1": "bowl", "bowl_2": "bowl",
        "water": "kettle", "coffee": "coffee_source",
        "stirrer": "spoon", "soup_tool_1": "spoon", "soup_tool_2": "spoon",
    }
    resolution = {
        "accepted": [
            {"generic_object_id": object_id, "semantic_label": label}
            for object_id, label in labels.items()
        ]
    }
    ledger = KitchenEffectLedger(KitchenGoalContract(("unused",), (), (), labels))
    runtime = SimpleNamespace(
        bundle=SimpleNamespace(resolution=resolution),
        ledger=ledger,
    )
    state = KitchenPlanningState(runtime)
    ledger.accept(
        (
            "poured(water,cup)", "poured(coffee,cup)", "stirred(stirrer,cup)",
            "placed(cup,serving_area)", "poured(water,mug)",
            "poured(coffee,mug)", "stirred(stirrer,mug)",
            "placed(mug,serving_area)", "placed(bowl_1,serving_area)",
            "placed(bowl_2,serving_area)", "placed(soup_tool_1,bowl_1)",
            "placed(soup_tool_2,bowl_2)",
        )
    )
    assert state.goal_verifier()


def test_the_executing_path_resolves_the_variant_from_the_flag_it_actually_gets():
    """Kitchen takes `--variant` when planning and `--physical-variant` when
    executing, and the two are mutually exclusive.

    A bare `arguments.variant` on the executing path is therefore always None.
    `load_expected(root, None)` fails with
    `TypeError: unsupported operand type(s) for /: 'PosixPath' and 'NoneType'`
    at the very end of an episode, after 20+ minutes of planning and physical
    manipulation, writing no artifact -- which is exactly what happened.

    No test exercises `main()` on the executing path, so this is asserted
    structurally: every call that consumes a variant outside a planning-only
    branch must resolve it from both flags.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "vlm_tamp_baseline/run_kitchen.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Names bound inside any `planning_only` branch: safe to use bare there.
    planning_only_scope: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "planning_only" in ast.dump(node.test):
            for sub in ast.walk(node):
                planning_only_scope.add(id(sub))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        if name not in {"load_expected", "write_execution_result"}:
            continue
        if id(node) in planning_only_scope:
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "variant"
                and getattr(sub.value, "id", "") == "arguments"
            ):
                offenders.append(name)
    assert not offenders, (
        f"{offenders} read arguments.variant on the executing path, where only "
        "--physical-variant is set; resolve with "
        "`arguments.variant or arguments.physical_variant`"
    )

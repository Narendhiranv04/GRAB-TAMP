from __future__ import annotations

from baseline_common.models import Entity, Observation, Region
from mujoco_scenes.tamp.baseline_observation_bridge import (
    BaselineRuntimeSnapshotObserver,
    _camera_visible_object_ids,
    observed_skill_precheck,
    observation_to_observed_state,
)
from mujoco_scenes.tamp.skills import FailureCode, SkillAction


def _observation(*, holding=None):
    return Observation(
        "kitchen",
        1,
        (
            Entity("mug", "object", "mug", {"region_id": "countertop"}),
            Entity("spoon", "object", "spoon", {"region_id": "countertop"}),
        ),
        (Region("countertop", "countertop", "open", True),),
        {"workspace": "home", "holding": holding},
    )


def test_camera_manifest_collects_only_positive_screen_evidence():
    manifest = {
        "cameras": {
            "left": {
                "objects": [
                    {"id": "mug", "pixel_count": 10},
                    {"id": "spoon", "pixel_count": 0},
                ]
            },
            "right": {"objects": [{"id": "mug", "pixel_count": 3}]},
        }
    }

    assert _camera_visible_object_ids(manifest) == frozenset({"mug"})


def test_camera_filter_hides_non_rendered_objects_but_keeps_held_object():
    state = observation_to_observed_state(
        _observation(holding="spoon"),
        camera_visible_object_ids={"mug"},
    )

    assert set(state.objects) == {"mug", "spoon"}
    assert state.robot.held_object == "spoon"


def test_camera_filter_removes_non_rendered_unheld_objects():
    state = observation_to_observed_state(
        _observation(),
        camera_visible_object_ids={"mug"},
    )

    assert set(state.objects) == {"mug"}


def test_snapshot_observer_persists_previously_segmented_objects():
    class Runtime:
        def __init__(self):
            self.calls = 0
            self.latest_annotation_manifest = {"cameras": {}}

        def observe(self):
            self.calls += 1
            visible = "mug" if self.calls == 1 else "spoon"
            self.latest_annotation_manifest = {
                "cameras": {"front": {"objects": [{"id": visible, "pixel_count": 9}]}}
            }
            return _observation(), ({"camera": "front", "data_url": "data:image/png;base64,eA=="},)

    observer = BaselineRuntimeSnapshotObserver(Runtime())
    first = observer()
    second = observer()

    assert set(first.state.objects) == {"mug"}
    assert set(second.state.objects) == {"mug", "spoon"}


def test_precheck_rejects_an_object_that_is_no_longer_visible():
    state = observation_to_observed_state(
        _observation(), camera_visible_object_ids={"mug"}
    )

    result = observed_skill_precheck(
        SkillAction("PICK", {"object_id": "spoon"}), state
    )

    assert result is not None
    assert result.failure_code is FailureCode.OBJECT_NOT_VISIBLE


def test_precheck_requires_the_observed_held_object_for_place():
    state = observation_to_observed_state(
        _observation(holding="spoon"), camera_visible_object_ids={"mug"}
    )

    result = observed_skill_precheck(
        SkillAction("PLACE", {"object_id": "mug", "region_id": "countertop"}),
        state,
    )

    assert result is not None
    assert result.failure_code is FailureCode.PRECONDITION_FAILED


def _kitchen_dialect_observation(*, held_object=None):
    """The robot dict the Kitchen runtime actually publishes.

    Kitchen reaches this bridge through `RobotObservation.as_dict()`, which
    spells the grasp `held_object` and the pose `location`, where the Living
    Room and Workshop runtimes spell them `holding` and `workspace`.  Every
    fixture above uses only the latter dialect, which is how the bridge came to
    read `holding` alone.
    """
    return Observation(
        "kitchen",
        1,
        (
            Entity("mug", "object", "mug", {"region_id": "countertop"}),
            Entity("spoon", "object", "spoon", {"region_id": "countertop"}),
        ),
        (Region("countertop", "countertop", "open", True),),
        {"location": "home", "held_object": held_object, "motion_ready": True},
    )


def test_bridge_reads_the_kitchen_held_object_dialect():
    state = observation_to_observed_state(
        _kitchen_dialect_observation(held_object="spoon")
    )
    assert state.robot.held_object == "spoon"
    assert state.robot.location == "home"


def test_precheck_admits_a_pour_the_kitchen_dialect_reports_as_held():
    """A pour after a successful pick must not be rejected as empty-handed.

    Reading `holding` alone made every Kitchen observation arrive with an empty
    gripper, so this precheck rejected every PLACE, POUR and STIR that followed
    a pick -- which is every one of them.
    """
    state = observation_to_observed_state(
        _kitchen_dialect_observation(held_object="spoon")
    )
    assert (
        observed_skill_precheck(
            SkillAction("POUR", {"source_id": "spoon", "target_id": "mug"}), state
        )
        is None
    )


def test_precheck_still_rejects_a_pour_with_a_genuinely_empty_gripper():
    state = observation_to_observed_state(_kitchen_dialect_observation())
    result = observed_skill_precheck(
        SkillAction("POUR", {"source_id": "spoon", "target_id": "mug"}), state
    )
    assert result is not None
    assert result.failure_code is FailureCode.PRECONDITION_FAILED

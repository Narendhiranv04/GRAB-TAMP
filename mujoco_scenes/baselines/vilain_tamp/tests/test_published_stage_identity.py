"""The model can only cite the stage id it was shown.

The observation payload publishes leak-safe stage ids while the stage
directories on disk keep the canonical region names.  Every consumer that
resolves a model-supplied stage reference therefore has to key on the published
spelling.  Keying on the canonical one cost 55 of 120 ViLaIn Kitchen episodes:
each detection citing an inspection stage failed with "detection references
unknown frame '004_region_0001'", the episode exited 1, and
`--continue-on-error` recorded it as a run.
"""

from __future__ import annotations

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    CameraFrameArtifacts,
    ViLaInObservation,
)
from mujoco_scenes.baselines.vilain_tamp.interpreter import _frame_index
from mujoco_scenes.baselines.vilain_tamp.observations import (
    prompt_observation_payload,
    public_stage_id_for,
)


def _frame(stage_id: str) -> CameraFrameArtifacts:
    return CameraFrameArtifacts(
        camera_id="overhead_camera",
        view_description="Overhead view",
        rgb_path=f"stages/{stage_id}/cameras/overhead_camera/rgb.png",
        depth_path=f"stages/{stage_id}/cameras/overhead_camera/depth.npy",
        calibration_path=f"stages/{stage_id}/cameras/overhead_camera/camera.json",
        rgb_sha256="a" * 64,
        depth_sha256="b" * 64,
        calibration_sha256="c" * 64,
    )


def _kitchen_inspection() -> ViLaInObservation:
    """An inspection of Kitchen D1, whose published alias is region_0004."""
    return ViLaInObservation(
        domain="kitchen",
        observation_mode="inspection",
        stage_id="004_d1",
        camera_frames=(_frame("004_d1"),),
        opened_region_id="D1",
        capture_timestamp="2026-09-09T00:00:00+00:00",
        inspection_ordinal=4,
        content_hash="d" * 64,
    )


def test_published_stage_id_hides_the_canonical_region():
    observation = _kitchen_inspection()
    published = public_stage_id_for(observation)
    assert published != observation.stage_id
    assert "d1" not in published
    assert "region_0004" in published


def test_the_payload_and_the_resolver_agree_on_the_stage_id():
    """Whatever the payload announces must be what the frame index accepts."""
    observation = _kitchen_inspection()
    payload = prompt_observation_payload((observation,))
    announced = payload["stages"][0]["stage_id"]
    assert announced == public_stage_id_for(observation)
    assert (announced, "overhead_camera") in _frame_index((observation,))


def test_the_canonical_stage_id_is_not_what_the_model_can_cite():
    """Guards the regression directly: keying on '004_d1' broke every
    detection, because the model is never shown that spelling."""
    observation = _kitchen_inspection()
    index = _frame_index((observation,))
    assert ("004_d1", "overhead_camera") not in index


def test_a_domain_without_aliases_is_unaffected():
    """Living Room publishes no aliases, so its ids pass through unchanged --
    which is why only Kitchen was losing episodes."""
    observation = ViLaInObservation(
        domain="living_room",
        observation_mode="initial_observation_only",
        stage_id="000_initial",
        camera_frames=(_frame("000_initial"),),
        opened_region_id=None,
        capture_timestamp="2026-09-09T00:00:00+00:00",
        inspection_ordinal=None,
        content_hash="d" * 64,
    )
    assert public_stage_id_for(observation) == "000_initial"
    assert ("000_initial", "overhead_camera") in _frame_index((observation,))

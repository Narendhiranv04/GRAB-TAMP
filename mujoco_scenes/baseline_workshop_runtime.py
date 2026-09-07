"""Physical Workshop execution for the comparison baselines.

The planning-only Workshop adapter in each baseline advances a private
symbolic rollout and never steps MuJoCo.  This module supplies the missing
half: it turns a baseline's ``Action`` into the generic Workshop operator the
ground-truth dispatcher already knows how to execute, runs it against a live
``WorkshopScene``, and reports what physically happened.

Two boundaries are deliberate and are the reason this file exists rather than
the baselines calling ``WorkshopExecutionDispatcher`` directly.

**The oracle's symbolic gate is not applied to a baseline.**
``WorkshopWorldState.check`` validates PLACE and SCREW against the solved
``WorkshopAssignment`` -- it rejects any object that is not part of "the
grounded repair pair" and any repair tuple that "does not match the grounded
assignment".  That gate encodes the answer.  Running a baseline through it
would reject a wrong-but-legal action symbolically instead of letting the
physics decide, and would score the baseline against privileged knowledge.
This module applies only domain-generic preconditions (is the hand free, is
the source open, is the object known) and lets the scene arbitrate the rest.

**Nothing derived from the oracle reaches the caller.**  Failure text is the
only channel by which an executor can leak: VLM-TAMP feeds action failures
back to the model as natural-language feedback.  Every ``ActionResult`` here
is built from a fixed vocabulary plus the caller's own anonymised identifiers,
never from dispatcher or state prose.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Protocol

from baseline_common.models import Action, ActionResult

from .workshop_ground_truth_execution import WorkshopExecutionDispatcher
from .workshop_ground_truth_planner import TARGET_JOINT, WorkshopAssignment
from .workshop_ground_truth_state import initial_workshop_state
from .workshop_scene import WorkshopScene


class WorkshopIdentifierSource(Protocol):
    """The subset of a baseline's planning runtime this module reads.

    Structural rather than a concrete import: ``mujoco_scenes`` must not depend
    on any baseline package, and each baseline owns its own anonymised
    identifier assignment.  Taking the mapping from the caller also guarantees
    the physical layer and the observation layer cannot drift apart.
    """

    internal_variant: str
    variant_spec: Mapping[str, Any]
    object_by_backend: Mapping[str, str]
    region_by_backend: Mapping[str, str]
    target_object_id: str


class _UngatedWorkshopState:
    """Stands in for the oracle's symbolic gate inside the dispatcher.

    ``WorkshopExecutionDispatcher.execute`` reads its ``state`` argument for
    exactly one purpose -- the ``state.check(action, assignment)`` call on its
    first line -- and never consults it again.  Substituting a permissive
    object therefore removes the assignment-aware gate without altering any
    physical behaviour, and this class exists to make that substitution
    explicit rather than implied.

    The preconditions a baseline must still satisfy are enforced by
    ``WorkshopPhysicalExecutor`` from its own bookkeeping.
    """

    @staticmethod
    def check(action: Mapping[str, Any], assignment: Any) -> tuple[bool, None]:
        return True, None


# Physical failure detail is summarised into these, so no dispatcher or
# world-state prose can reach a model through failure feedback.
_PHYSICAL_FAILURE_CODE = "physical_execution_failed"


class WorkshopPhysicalExecutor:
    """Execute baseline Workshop actions against a live MuJoCo scene."""

    def __init__(
        self,
        runtime: WorkshopIdentifierSource,
        *,
        # Tracks what `run_workshop_ground_truth_execution` constructs, so a
        # baseline is held to exactly the oracle's standard and never a harder
        # one.  Workshop runs the assisted grasp path; contact-gated execution
        # was measured and does not currently pass (see BASELINE_FIDELITY.md).
        strict_physical_execution: bool = False,
        frame_callback: Callable[[bool], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.scene = WorkshopScene(
            robot="google", variant=str(runtime.internal_variant)
        )
        self.state = initial_workshop_state(
            dict(runtime.variant_spec["storage_contents"])
        )
        # The baseline has not committed to a repair pair yet, and the
        # executor must not assume one.  `driver`/`fastener` are filled in
        # from the baseline's own choices as it makes them; the dispatcher
        # uses them only to widen allowed-contact sets and to pick a grasp
        # offset, never to decide whether an action succeeds.
        self._assignment = WorkshopAssignment(
            variant_id=str(runtime.internal_variant),
            intended_outcome="UNKNOWN",
            is_feasible=True,
            driver=None,
            fastener=None,
            work_surface=None,
            parts_container=None,
            target_joint=TARGET_JOINT,
            assignment_source="BASELINE_SELECTED",
        )
        self.dispatcher = WorkshopExecutionDispatcher(
            self.scene,
            self._assignment,
            frame_callback=frame_callback,
            strict_physical_execution=strict_physical_execution,
        )
        self._ungated_state = _UngatedWorkshopState()
        self.executed_actions = 0
        self.direct_payload_pose_writes = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Identifier translation
    # ------------------------------------------------------------------
    def _backend_for_object(self, object_id: str) -> str | None:
        return self.runtime.object_by_backend.get(object_id)

    def _backend_for_region(self, region_id: str) -> str | None:
        return self.runtime.region_by_backend.get(region_id)

    def _backend_for_destination(self, destination_id: str) -> str | None:
        """Resolve a destination that may name a region or the frame joint."""
        region = self._backend_for_region(destination_id)
        if region is not None:
            return region
        return self._backend_for_object(destination_id)

    # ------------------------------------------------------------------
    # Physical dispatch
    # ------------------------------------------------------------------
    def _run(self, operator: str, arguments: list[str]) -> dict[str, Any]:
        action = {"operator": operator, "arguments": arguments}
        self.dispatcher.assignment = self._assignment
        try:
            result = self.dispatcher.execute(action, self._ungated_state)
        except RuntimeError as error:
            # The ground-truth runner lets these abort the episode, because a
            # plan the oracle produced should never be physically impossible.
            # A baseline's plan can be, and routinely will be: an unreachable
            # grasp is an ordinary wrong answer, not a crash.  The message is
            # kept for the trace only -- it names backend bodies and geoms, so
            # it must not reach a caller that forwards text to a model.
            result = {
                "success": False,
                "status": "PHYSICAL_MOTION_FAILED",
                "operator": operator,
                "arguments": arguments,
                "exception": repr(error),
                "direct_payload_pose_write": False,
            }
        self.executed_actions += 1
        if result.get("direct_payload_pose_write"):
            self.direct_payload_pose_writes += 1
        self.history.append({"action": action, "physical_result": result})
        return result

    @staticmethod
    def _physical_failure(operator: str) -> ActionResult:
        # Deliberately free of any detail from the dispatcher: the caller may
        # forward this text to a model.
        return ActionResult.failed(
            _PHYSICAL_FAILURE_CODE,
            f"The robot could not complete {operator.lower()} in the scene.",
        )

    # ------------------------------------------------------------------
    def execute(self, action: Action) -> ActionResult:
        skill = action.skill.upper()
        args = action.arguments
        if skill == "INSPECT":
            return self._inspect(str(args.get("region_id", "")))
        if skill == "PICK":
            return self._pick(str(args.get("object_id", "")))
        if skill == "PLACE":
            return self._place(
                str(args.get("object_id", "")), str(args.get("region_id", ""))
            )
        if skill == "INSERT":
            return self._insert(
                str(args.get("fastener_id", "")), str(args.get("target_id", ""))
            )
        if skill == "FASTEN":
            return self._fasten(
                str(args.get("tool_id", "")),
                str(args.get("fastener_id", "")),
                str(args.get("target_id", "")),
            )
        return ActionResult.failed(
            "unsupported_planning_action",
            f"Workshop cannot apply {action.skill}",
            recoverable=False,
        )

    # ------------------------------------------------------------------
    def _inspect(self, region_id: str) -> ActionResult:
        backend = self._backend_for_region(region_id)
        if backend is None or backend not in self.state.storage_open:
            return ActionResult.failed(
                "unknown_region", f"Unknown storage region {region_id}"
            )
        if self.state.held_object is not None:
            return ActionResult.failed(
                "gripper_occupied", "Inspect requires an empty gripper"
            )
        if self.state.storage_open[backend]:
            # Opening is also the inspection action, and a region stays open,
            # so re-inspecting is a no-op rather than a physical failure.
            return ActionResult.succeeded(f"inspected({region_id})")
        result = self._run("OPEN", [backend])
        if not result.get("success"):
            return self._physical_failure("INSPECT")
        self.state.storage_open[backend] = True
        self.state.inspected_storage.add(backend)
        return ActionResult.succeeded(f"inspected({region_id})")

    def _pick(self, object_id: str) -> ActionResult:
        backend = self._backend_for_object(object_id)
        if backend is None:
            return ActionResult.failed(
                "unknown_object", f"Object {object_id} is not observed"
            )
        if object_id == self.runtime.target_object_id:
            return ActionResult.failed(
                "fixed_target", "The frame joint is not movable"
            )
        if self.state.held_object is not None:
            held = self.runtime.object_by_backend.get(
                self.state.held_object, self.state.held_object
            )
            return ActionResult.failed(
                "gripper_occupied", f"The gripper already holds {held}"
            )
        source = self.state.object_locations.get(backend)
        if source is None:
            return ActionResult.failed(
                "unknown_object", f"Object {object_id} is not observed"
            )
        if source in self.state.storage_open and not self.state.storage_open[source]:
            return ActionResult.failed(
                "source_closed",
                f"{object_id} is inside a storage region that is not open",
            )
        result = self._run("PICK", [backend, source])
        if not result.get("success"):
            return self._physical_failure("PICK")
        self.state.held_object = backend
        self.state.object_locations[backend] = "GRIPPER"
        return ActionResult.succeeded(f"holding({object_id})")

    def _place(self, object_id: str, destination_id: str) -> ActionResult:
        backend = self._backend_for_object(object_id)
        destination = self._backend_for_destination(destination_id)
        if backend is None or self.state.held_object != backend:
            return ActionResult.failed(
                "not_holding_object", f"The robot is not holding {object_id}"
            )
        if destination is None:
            return ActionResult.failed(
                "unknown_destination", f"Unknown destination {destination_id}"
            )
        if destination == TARGET_JOINT:
            # PLACE onto the frame joint is an insertion; route it so the
            # dispatcher selects the insertion primitive.
            return self._insert(object_id, destination_id)
        result = self._run("PLACE", [backend, destination])
        if not result.get("success"):
            return self._physical_failure("PLACE")
        self.state.held_object = None
        self.state.object_locations[backend] = destination
        return ActionResult.succeeded(f"placed({object_id},{destination_id})")

    def _insert(self, fastener_id: str, target_id: str) -> ActionResult:
        backend = self._backend_for_object(fastener_id)
        target = self._backend_for_destination(target_id)
        if backend is None or self.state.held_object != backend:
            return ActionResult.failed(
                "not_holding_fastener", "Insert requires the held fastener"
            )
        if target != TARGET_JOINT:
            return ActionResult.failed(
                "invalid_insertion", f"{target_id} does not accept a fastener"
            )
        # Record the baseline's choice before dispatch: the insertion
        # primitive reads it when widening allowed contacts.  Whether this
        # screw actually fits is settled by the scene, not by this field.
        self._assignment = replace(self._assignment, fastener=backend)
        result = self._run("PLACE", [backend, TARGET_JOINT])
        if not result.get("success"):
            return self._physical_failure("INSERT")
        self.state.held_object = None
        self.state.object_locations[backend] = TARGET_JOINT
        self.state.inserted_fastener = (backend, TARGET_JOINT)
        return ActionResult.succeeded(f"inserted({fastener_id},{target_id})")

    def _fasten(
        self, tool_id: str, fastener_id: str, target_id: str
    ) -> ActionResult:
        tool = self._backend_for_object(tool_id)
        fastener = self._backend_for_object(fastener_id)
        target = self._backend_for_destination(target_id)
        if tool is None or self.state.held_object != tool:
            return ActionResult.failed(
                "not_holding_tool", "Fasten requires the held driver"
            )
        if fastener is None or target != TARGET_JOINT:
            return ActionResult.failed(
                "invalid_driver", "The named parts cannot fasten this joint"
            )
        if self.state.inserted_fastener != (fastener, TARGET_JOINT):
            return ActionResult.failed(
                "fastener_not_inserted",
                f"{fastener_id} is not inserted in {target_id}",
            )
        self._assignment = replace(self._assignment, driver=tool)
        result = self._run("SCREW", [tool, fastener, TARGET_JOINT])
        if not result.get("success"):
            return self._physical_failure("FASTEN")
        self.state.repaired_joint = TARGET_JOINT
        return ActionResult.succeeded(
            f"fastened({tool_id},{fastener_id},{target_id})"
        )

    # ------------------------------------------------------------------
    @property
    def goal_satisfied(self) -> bool:
        """True once the joint has been physically repaired."""
        return self.state.repaired_joint == TARGET_JOINT

    def close(self) -> None:
        recorder_close = getattr(self.dispatcher, "close", None)
        if callable(recorder_close):
            recorder_close()


class MirroredWorkshopExecutor:
    """Physical execution that keeps a planning runtime's view in step.

    A baseline's planning runtime renders and textualises from its own closed
    model and private symbolic state, not from the live physical scene.  If
    physical execution replaced the symbolic rollout outright, the model would
    keep seeing the initial world no matter what the robot did -- drawers would
    read as shut after being opened, and a picked object would still appear in
    storage.

    So the physical scene decides the outcome and the symbolic rollout is
    replayed only to keep observations current.  The mirror is never allowed to
    overturn the physical verdict: its own result is recorded for auditing and
    otherwise discarded.
    """

    def __init__(self, physical: WorkshopPhysicalExecutor, mirror: Any) -> None:
        self.physical = physical
        self.mirror = mirror
        # A physical success the observation mirror rejects means the two
        # models of the world have diverged.  It does not change what is
        # reported, but it is worth being able to find afterwards.
        self.mirror_divergences: list[dict[str, Any]] = []

    def execute(self, action: Action) -> ActionResult:
        result = self.physical.execute(action)
        if result.success:
            mirrored = self.mirror.execute(action)
            if not mirrored.success:
                self.mirror_divergences.append({
                    "action": action.as_dict(),
                    "mirror_failure_code": mirrored.failure_code,
                })
        return result

    @property
    def goal_satisfied(self) -> bool:
        return self.physical.goal_satisfied

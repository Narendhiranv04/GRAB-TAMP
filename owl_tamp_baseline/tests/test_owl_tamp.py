from __future__ import annotations

import base64
import random
from collections import deque

from baseline_common.models import ActionResult, Entity, Observation, Region

from owl_tamp_baseline.domain import executed_encoding, relaxed_ground
from owl_tamp_baseline.models import Action, PlanSketch, PlanningResult
from owl_tamp_baseline.planner import (
    OWLTAMPPlanner,
    OWLTAMPPlannerConfig,
    protocol_max_tokens,
)
from owl_tamp_baseline.receding_horizon import OWLTAMPRecedingHorizon
from owl_tamp_baseline.replanning import OWLTAMPReplanning
from owl_tamp_baseline.refinement import (
    appendix_plan_modification,
    constrained_breadth_first_search,
    search_then_sample,
)


def observation() -> Observation:
    return Observation(
        "living_room",
        0,
        (Entity("object_0001", "object", "object_0001", {"region_id": "staging_area"}),),
        (
            Region("staging_area", "staging_area", "open", True),
            Region("region_0001", "region_0001", "open", True),
        ),
        {"holding": None, "workspace": "home"},
        False,
    )


def test_relaxed_grounding_and_executed_encoding() -> None:
    grounded = relaxed_ground(
        "living_room", {"object_0001"}, {"staging_area", "region_0001"}
    )
    pick = Action("PICK", ("object_0001",))
    place = Action("PLACE", ("object_0001", "region_0001"))
    assert pick in grounded and place in grounded
    rows = executed_encoding((pick, place))
    assert rows[0]["extra_precondition"] == "Executed(0)"
    assert rows[1]["extra_effect"] == "Executed(2)"


def test_workshop_relaxed_grounding_supports_inspection_and_fastening() -> None:
    """The repaired fixture must be reachable as an INSERT/FASTEN target.

    Regression: the joint was passed only as "not movable", which removed it
    from every argument position.  `INSERT(screw, joint)` and
    `FASTEN(driver, screw, joint)` -- the only actions that can satisfy the
    Workshop goal -- were then absent from the grounded set, so the model chose
    targets from a menu without the right answer, and `validate_sketch` would
    have rejected the right answer anyway.
    """
    grounded = relaxed_ground(
        "workshop",
        {"object_0002", "object_0003"},              # movable: driver, screw
        {"region_0001", "region_0004"},
        {"region_0001"},
        fixed_object_ids={"object_0001"},            # the frame joint
    )
    assert Action("INSPECT", ("region_0001",)) in grounded
    assert Action("FASTEN", ("object_0002", "object_0003", "object_0001")) in grounded
    assert Action("INSERT", ("object_0003", "object_0001")) in grounded
    assert Action("PICK", ("object_0001",)) not in grounded
    # INSERT is the only way to seat a fastener.  Grounding PLACE onto the
    # fixture too would give the domain two spellings that the physical layer
    # treats identically and the transition model does not: PLACE records
    # `at(...)`, while FASTEN's precondition needs `inserted(...)`, so a plan
    # could place onto the joint and then never be able to fasten it.
    assert Action("PLACE", ("object_0003", "object_0001")) not in grounded
    # A movable object is never a fastening target.
    assert Action("FASTEN", ("object_0002", "object_0003", "object_0003")) not in grounded


def test_grounding_is_unchanged_when_no_fixture_is_declared() -> None:
    """Kitchen and Living Room pass no movable subset, so they see no change,
    and neither does single-shot Workshop, whose pools are empty while the
    storage is still closed.  This pins the already-reported grid.
    """
    regions = {"region_0001", "region_0002", "region_0003", "region_0004"}
    for scene in ("kitchen", "living_room"):
        objects = {"object_0001", "object_0002"}
        assert set(relaxed_ground(scene, objects, regions)) == set(
            relaxed_ground(scene, objects, regions, (), fixed_object_ids=set())
        )
    # Single-shot Workshop: nothing inspected, so nothing is movable.
    closed = {"region_0001", "region_0002", "region_0003"}
    assert set(relaxed_ground("workshop", set(), regions, closed)) == set(
        relaxed_ground("workshop", set(), regions, closed, fixed_object_ids={"object_0001"})
    )


def test_single_call_protocol_reserves_prompt_context() -> None:
    assert protocol_max_tokens(8192, "single_call") == 2048
    assert protocol_max_tokens(8192, "native") == 8192


def test_symbolic_search_requires_goal_literals() -> None:
    state = observation()
    grounded = relaxed_ground("living_room", state.object_ids, state.region_ids)
    plan = constrained_breadth_first_search(
        state,
        grounded,
        (Action("PLACE", ("object_0001", "region_0001")),),
        ("at(object_0001,region_0001)",),
    )
    assert plan == (
        Action("PICK", ("object_0001",)),
        Action("PLACE", ("object_0001", "region_0001")),
    )


def test_symbolic_search_handles_full_living_room_plan_with_default_budget() -> None:
    objects = tuple(f"object_{index:04d}" for index in range(1, 6))
    regions = ("staging_area", "region_0001", "region_0002", "region_0003")
    state = Observation(
        "living_room",
        0,
        tuple(
            Entity(item, "object", item, {"region_id": "staging_area"})
            for item in objects
        ),
        tuple(Region(item, item, "open", True) for item in regions),
        {"holding": None, "workspace": "home"},
        False,
    )
    targets = ("region_0001", "region_0001", "region_0002", "region_0003", "region_0003")
    sketch = tuple(
        action
        for object_id, target in zip(objects, targets)
        for action in (
            Action("PICK", (object_id,)),
            Action("PLACE", (object_id, target)),
        )
    )
    grounded = relaxed_ground("living_room", state.object_ids, state.region_ids)
    goals = tuple(
        f"at({object_id},{target})"
        for object_id, target in zip(objects, targets)
    )

    assert constrained_breadth_first_search(state, grounded, sketch, goals) == sketch


class FakeTransport:
    def __init__(self):
        self.payloads = []
        self.responses = deque(
            [
                {
                    "status": "PLAN",
                    "actions": [
                        {"operator": "PICK", "arguments": ["object_0001"]},
                        {"operator": "PLACE", "arguments": ["object_0001", "region_0001"]},
                    ],
                    "goal_literals": ["at(object_0001,region_0001)"],
                },
                {
                    "constraints": [
                        {"action_index": 0, "description": "reachable grasp", "expression": "reachable(0)"}
                    ]
                },
                {
                    "constraints": [
                        {"action_index": 1, "description": "stable support", "expression": "supported_by(object_0001, region_0001)"}
                    ]
                },
            ]
        )

    def complete(self, payload):
        self.payloads.append(payload)
        content = self.responses.popleft()
        return {"choices": [{"message": {"content": content}}]}


def test_two_stage_planner_uses_images_and_refines() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
    transport = FakeTransport()
    planner = OWLTAMPPlanner(
        OWLTAMPPlannerConfig(), transport=transport
    )
    result = planner.plan(
        "Put the visible object on the target support",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
    )
    assert result.status == "PLAN"
    assert len(result.actions) == 2
    assert result.samples_tested == 2
    assert not transport.responses
    assert len(transport.payloads) == 3
    sketch_payload = transport.payloads[0]
    sketch_format = sketch_payload["response_format"]
    assert sketch_format["type"] == "json_schema"
    assert sketch_format["json_schema"]["strict"] is True
    sketch_text = sketch_payload["messages"][0]["content"][0]["text"]
    assert "Return no more than 64 actions" in sketch_text
    assert "only task-essential action choices" in sketch_text
    assert "Do not enumerate" in sketch_text
    assert "Format example only" not in sketch_text
    assert "example_object" not in sketch_text
    constraint_format = transport.payloads[1]["response_format"]
    constraint = constraint_format["json_schema"]["schema"]
    assert constraint["properties"]["constraints"]["maxItems"] == 1
    assert constraint["properties"]["constraints"]["items"]["properties"][
        "action_index"
    ]["const"] == 0
    assert "pattern" in constraint["properties"]["constraints"]["items"][
        "properties"
    ]["expression"]
    assert len(planner.response_trace) == 3


def test_planner_can_exclude_fixed_scene_objects_from_relaxed_grounding() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")

    class NoPlanTransport:
        def complete(self, _payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": {
                                "status": "NO_PLAN",
                                "actions": [],
                                "goal_literals": [],
                            }
                        }
                    }
                ]
            }

    planner = OWLTAMPPlanner(OWLTAMPPlannerConfig(), transport=NoPlanTransport())
    planner.plan(
        "Inspect the available storage",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
        movable_object_ids=(),
    )

    prompt = planner.trace["model_prompts"]["discrete_sketch"]
    assert '"operator":"PICK"' not in prompt
    assert '"operator":"PLACE"' not in prompt


def test_single_call_protocol_skips_auxiliary_constraint_queries() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")

    class SketchOnlyTransport:
        def __init__(self):
            self.calls = 0

        def complete(self, _payload):
            self.calls += 1
            return {
                "choices": [{"message": {"content": {
                    "status": "PLAN",
                    "actions": [
                        {"operator": "PICK", "arguments": ["object_0001"]},
                        {"operator": "PLACE", "arguments": ["object_0001", "region_0001"]},
                    ],
                    "goal_literals": ["at(object_0001,region_0001)"],
                }}}]
            }

    transport = SketchOnlyTransport()
    planner = OWLTAMPPlanner(OWLTAMPPlannerConfig(), transport=transport)
    result = planner.plan(
        "Put the object on the target support",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
        max_vlm_requests=1,
    )

    assert transport.calls == 1
    assert len(planner.response_trace) == 1
    assert planner.trace["constraint_generation_complete"] is False
    assert result.status == "PLAN"


class InvalidConstraintTransport:
    def __init__(self):
        self.responses = deque(
            [
                {
                    "status": "PLAN",
                    "actions": [
                        {"operator": "PICK", "arguments": ["object_0001"]},
                    ],
                    "goal_literals": ["holding(object_0001)"],
                },
                {
                    "constraints": [
                        {
                            "action_index": 0,
                            "description": "unsupported model helper",
                            "expression": "graspable(object_0001)",
                        }
                    ]
                },
            ]
        )

    def complete(self, _payload):
        content = self.responses.popleft()
        return {"choices": [{"message": {"content": content}}]}


def test_invalid_model_constraint_is_a_scored_failure_not_a_crash() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
    planner = OWLTAMPPlanner(
        OWLTAMPPlannerConfig(), transport=InvalidConstraintTransport()
    )
    result = planner.plan(
        "Pick the visible object",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
    )
    assert result.status == "INVALID_MODEL_OUTPUT"
    assert not result.actions
    assert "unknown helper" in result.failure


class InvalidJSONTransport:
    def complete(self, _payload):
        return {"choices": [{"message": {"content": '{"status": "PLAN"'}}]}


def test_invalid_json_is_a_scored_failure_not_a_crash() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
    planner = OWLTAMPPlanner(
        OWLTAMPPlannerConfig(), transport=InvalidJSONTransport()
    )
    result = planner.plan(
        "Pick the visible object",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
    )
    assert result.status == "INVALID_MODEL_OUTPUT"
    assert result.failure == (
        "Invalid OWL-TAMP discrete sketch: Completion content is not valid JSON"
    )


def test_receding_horizon_reobserves_after_each_successful_action() -> None:
    state = {"revision": 0, "held": False, "done": False}

    class Planner:
        response_trace = [{"request": 1}, {"request": 2}]
        trace = {"stage": "initial"}

        def plan(self, _goal, observation, _images, _oracle, **_kwargs):
            if observation.revision == 0:
                action = Action("PICK", ("object_0001",))
            else:
                action = Action("PLACE", ("object_0001", "region_0001"))
            return PlanningResult("PLAN", PlanSketch("PLAN", (action,), ()), (action,), (), 1, 1)

    def observe():
        return (
            Observation(
                "living_room",
                state["revision"],
                (Entity("object_0001", "object", "object", {}),),
                (Region("region_0001", "region", "open", True),),
                {"holding": state["held"]},
                state["done"],
            ),
            (),
        )

    def execute(action):
        if action.operator == "PICK":
            state["held"] = True
        else:
            state["held"] = False
            state["done"] = True
        state["revision"] += 1
        return ActionResult.succeeded(f"completed({action.operator})")

    result = OWLTAMPRecedingHorizon(
        Planner(), observe, execute, lambda row: row.goal_satisfied,
        lambda _action, _constraints, _trial: True,
    ).run("Place the object")

    assert result.success
    assert result.planning_rounds == 2
    assert result.replans == 1
    assert result.executed_actions == 2
    assert result.raw_vlm_requests == 4


class _WorkshopStubPlanner:
    """Plans inspection first, then the repair the inspection revealed.

    Stands in for the real Workshop behaviour: nothing inside the storage can
    be named until a fresh observation shows it.
    """

    trace = {"stage": "stub"}

    def __init__(self, calls_per_cycle: int = 4) -> None:
        self.calls_per_cycle = calls_per_cycle
        self.response_trace: list[dict[str, int]] = []
        self.seen_kwargs: list[dict[str, object]] = []

    def plan(self, _goal, observation, _images, _oracle, **kwargs):
        self.seen_kwargs.append(kwargs)
        budget = kwargs.get("max_vlm_requests")
        spend = self.calls_per_cycle if budget is None else min(self.calls_per_cycle, budget)
        self.response_trace = [{"request": index} for index in range(spend)]
        if not any(entity.entity_id == "screw" for entity in observation.entities):
            actions = (Action("INSPECT", ("drawer",)),)
        else:
            actions = (Action("PICK", ("screw",)), Action("PLACE", ("screw", "joint")))
        return PlanningResult("PLAN", PlanSketch("PLAN", actions, ()), actions, (), 1, 1)


def _workshop_world():
    state = {"revision": 0, "inspected": False, "done": False, "held": None}

    def observe():
        entities = ()
        if state["inspected"]:
            entities = (Entity("screw", "object", "screw", {"region_id": "drawer"}),)
        return (
            Observation(
                "workshop",
                state["revision"],
                entities,
                (Region("drawer", "drawer", "open" if state["inspected"] else "closed", state["inspected"]),),
                {"holding": state["held"]},
                state["done"],
            ),
            (),
        )

    def execute(action):
        state["revision"] += 1
        if action.operator == "INSPECT":
            state["inspected"] = True
        elif action.operator == "PICK":
            state["held"] = action.arguments[0]
        else:
            state["held"] = None
            state["done"] = True
        return ActionResult.succeeded(f"completed({action.operator})")

    return state, observe, execute


def test_replanning_reaches_a_goal_that_needs_a_second_observation() -> None:
    _state, observe, execute = _workshop_world()
    planner = _WorkshopStubPlanner()

    result = OWLTAMPReplanning(
        planner, observe, execute, lambda row: row.goal_satisfied,
        lambda _action, _constraints, _trial: True,
        max_model_calls=15,
    ).run("Repair the joint")

    assert result.success and result.status == "GOAL_COMPLETE"
    # Cycle one can only inspect; cycle two sees the screw and finishes.
    assert result.planning_cycles == 2
    assert result.replans == 1
    assert result.executed_actions == 3
    assert result.clean_cycles == 2
    assert result.model_calls == 8


def test_replanning_spends_one_total_model_budget_across_cycles() -> None:
    _state, observe, execute = _workshop_world()
    planner = _WorkshopStubPlanner(calls_per_cycle=6)

    result = OWLTAMPReplanning(
        planner, observe, execute, lambda row: row.goal_satisfied,
        lambda _action, _constraints, _trial: True,
        max_model_calls=10,
    ).run("Repair the joint")

    # The budget is a whole-episode one: the second cycle is offered only what
    # the first left behind, and no cycle may exceed the episode total.
    assert [row["max_vlm_requests"] for row in planner.seen_kwargs] == [10, 4]
    assert result.model_calls <= 10


def test_replanning_stops_when_the_model_budget_runs_out() -> None:
    class Stalled(_WorkshopStubPlanner):
        def plan(self, _goal, observation, _images, _oracle, **kwargs):
            budget = kwargs.get("max_vlm_requests")
            self.response_trace = [{"request": index} for index in range(min(4, budget))]
            actions = (Action("INSPECT", ("drawer",)),)
            return PlanningResult("PLAN", PlanSketch("PLAN", actions, ()), actions, (), 1, 1)

    state = {"revision": 0}

    def observe():
        state["revision"] += 1
        return (
            Observation(
                "workshop", state["revision"], (),
                (Region("drawer", "drawer", "closed", False),), {"holding": None}, False,
            ),
            (),
        )

    result = OWLTAMPReplanning(
        Stalled(), observe, lambda _action: ActionResult.succeeded("inspected"),
        lambda row: row.goal_satisfied, lambda _a, _c, _t: True,
        max_model_calls=9, max_stagnant_cycles=16,
    ).run("Repair the joint")

    assert not result.success
    assert result.status == "MODEL_BUDGET_EXHAUSTED"
    assert result.model_calls == 9
    assert result.planning_cycles == 3


def test_unobserved_goal_literal_ends_a_single_shot_cycle_but_not_a_replanning_one() -> None:
    sketch = PlanSketch(
        "PLAN",
        (Action("PICK", ("object_0001",)),),
        # Names an object no observation has revealed -- what the Workshop
        # model emits at cycle one, when the storage is still shut.
        ("at(unseen_screw,region_0001)",),
    )
    grounded = relaxed_ground(
        "living_room", {"object_0001"}, {"staging_area", "region_0001"}
    )
    strict = search_then_sample(
        observation(), grounded, sketch, (), lambda _a, _c, _t: True
    )
    assert strict.status == "INVALID_GOAL_LITERALS"

    lenient = search_then_sample(
        observation(), grounded, sketch, (), lambda _a, _c, _t: True,
        drop_unobserved_goals=True,
    )
    assert lenient.status == "PLAN"
    assert lenient.dropped_goal_literals == ("at(unseen_screw,region_0001)",)
    assert Action("PICK", ("object_0001",)) in lenient.actions


def test_search_does_not_invent_gap_fillers_the_refiner_cannot_accept() -> None:
    """Regression: a Workshop cycle built a 13-action skeleton containing
    `PLACE(manual_screwdriver, power_screwdriver)` purely to free the gripper,
    then spent its whole 500-sample budget proving it impossible -- wasting the
    ten model calls that cycle had cost.  Relaxed grounding offers every object
    as a destination and the transition model does not check, so only the
    oracle knows the placement is inadmissible.
    """
    state = Observation(
        "workshop",
        0,
        (
            Entity("object_0001", "object", "object_0001", {"region_id": "region_0001"}),
            Entity("object_0002", "object", "object_0002", {"region_id": "region_0001"}),
        ),
        (
            Region("region_0001", "region_0001", "open", True),
            Region("region_0004", "region_0004", "open", True),
        ),
        {"holding": None, "workspace": "workbench"},
        False,
    )
    grounded = relaxed_ground(
        "workshop", {"object_0001", "object_0002"}, {"region_0001", "region_0004"}
    )
    # Relaxed grounding offers every object as a PLACE destination, which is
    # where the inadmissible filler comes from.
    assert Action("PLACE", ("object_0001", "object_0002")) in grounded

    # Only regions are real destinations in this scene.
    def admissible(action):
        if action.operator == "PLACE":
            return action.arguments[1] in {"region_0001", "region_0004"}
        return True

    # A sketch whose second PICK needs the gripper freed first.
    sketch = (
        Action("PICK", ("object_0001",)),
        Action("PICK", ("object_0002",)),
    )
    loose = constrained_breadth_first_search(state, grounded, sketch)
    assert loose is not None
    assert any(
        step.operator == "PLACE" and not admissible(step) for step in loose
    ), "expected the unpruned search to invent an inadmissible placement"

    pruned = constrained_breadth_first_search(
        state, grounded, sketch, admissible=admissible
    )
    assert pruned is not None
    assert all(admissible(step) for step in pruned)
    # The sketch is still contained in the plan, in order.
    assert [s for s in pruned if s in sketch] == list(sketch)


def test_appendix_modification_relocates_a_blocking_object() -> None:
    state = Observation(
        "living_room",
        0,
        (
            Entity("object_0001", "object", "object_0001", {"region_id": "staging_area"}),
            Entity("object_0002", "object", "object_0002", {"region_id": "region_0001"}),
        ),
        (
            Region("staging_area", "staging_area", "open", True),
            Region("region_0001", "region_0001", "open", True),
        ),
        {"holding": None, "workspace": "home"},
        False,
    )
    skeleton = (
        Action("PICK", ("object_0001",)),
        Action("PLACE", ("object_0001", "region_0001")),
    )
    modified = appendix_plan_modification(
        state, skeleton, skeleton, 1, random.Random(0)
    )
    assert modified is not None
    # The blocking occupant is moved out before the robot picks up the object
    # it is trying to place -- a PICK is not applicable with a full gripper.
    assert modified[0] == Action("PICK", ("object_0002",))
    assert modified[1].operator == "PLACE" and modified[1].arguments[0] == "object_0002"
    assert modified[1].arguments[1] != "region_0001"
    assert modified[2:] == skeleton


def test_appendix_modification_declines_when_nothing_blocks() -> None:
    skeleton = (
        Action("PICK", ("object_0001",)),
        Action("PLACE", ("object_0001", "region_0001")),
    )
    assert appendix_plan_modification(
        observation(), skeleton, skeleton, 1, random.Random(0)
    ) is None


def test_backtracking_is_off_unless_asked_for() -> None:
    state = Observation(
        "living_room",
        0,
        (
            Entity("object_0001", "object", "object_0001", {"region_id": "staging_area"}),
            Entity("object_0002", "object", "object_0002", {"region_id": "region_0001"}),
        ),
        (
            Region("staging_area", "staging_area", "open", True),
            Region("region_0001", "region_0001", "open", True),
        ),
        {"holding": None, "workspace": "home"},
        False,
    )
    sketch = PlanSketch(
        "PLAN",
        (
            Action("PICK", ("object_0001",)),
            Action("PLACE", ("object_0001", "region_0001")),
        ),
        (),
    )
    grounded = relaxed_ground(
        "living_room", {"object_0001", "object_0002"}, {"staging_area", "region_0001"}
    )

    # An oracle that refuses the contested placement until the occupant moves.
    def oracle(action, _constraints, _trial):
        if action.operator == "PLACE" and action.arguments == ("object_0001", "region_0001"):
            return "object_0002" in relocated
        if action.operator == "PLACE" and action.arguments[0] == "object_0002":
            relocated.add("object_0002")
        return True

    relocated: set[str] = set()
    single = search_then_sample(
        state, grounded, sketch, (), oracle, max_samples_per_action=3
    )
    assert single.status == "NO_CONTINUOUS_PLAN"
    assert single.skeletons_tested == 1

    relocated = set()
    backtracked = search_then_sample(
        state, grounded, sketch, (), oracle,
        max_samples_per_action=3, backtrack=True, rng=random.Random(0),
    )
    assert backtracked.status == "PLAN"
    assert backtracked.skeletons_tested == 2
    assert Action("PICK", ("object_0002",)) in backtracked.actions

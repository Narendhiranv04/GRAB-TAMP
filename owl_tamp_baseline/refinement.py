"""Executed(i) symbolic search followed by bounded continuous sampling."""

from __future__ import annotations

import heapq
import itertools
import random

from collections import deque
from dataclasses import dataclass
import re
from typing import Callable, Mapping, Protocol, Sequence

from baseline_common.models import Observation

from .models import Action, Constraint, PlanningResult, PlanSketch


PAPER_MAX_SAMPLES_PER_ACTION = 500
PAPER_MAX_SKELETONS = 5


class SampleOracle(Protocol):
    def __call__(
        self,
        action: Action,
        constraints: Sequence[Constraint],
        sample_index: int,
    ) -> bool: ...


@dataclass(frozen=True)
class _State:
    holding: str | None
    locations: tuple[tuple[str, str], ...]
    opened: frozenset[str]
    facts: frozenset[str]
    executed: int

    @property
    def location_map(self) -> dict[str, str]:
        return dict(self.locations)


def _initial_state(observation: Observation) -> _State:
    holding = observation.robot.get("holding")
    holding = str(holding) if holding else None
    locations = []
    for entity in observation.entities:
        location = (
            entity.facts.get("region_id")
            or entity.facts.get("location")
            # Kitchen publishes an object's region as `source_region`; the
            # Living Room and the Workshop publish `region_id`.  Reading only
            # the latter two left every Kitchen object with no location, so
            # PICK was never applicable and the search dead-ended after 32
            # expansions with only the region OPENs reachable -- which is the
            # whole of OWL-TAMP's 0.0% Kitchen result.  Same dialect split as
            # the `holding`/`held_object` bridge bug.
            or entity.facts.get("source_region")
        )
        if location and location != "held":
            locations.append((entity.entity_id, str(location)))
    opened = frozenset(
        region.region_id for region in observation.regions if region.state == "open"
    )
    facts = {f"at({object_id},{region_id})" for object_id, region_id in locations}
    facts.update(f"open({region_id})" for region_id in opened)
    if holding:
        facts.add(f"holding({holding})")
    return _State(holding, tuple(sorted(locations)), opened, frozenset(facts), 0)


def _transition(state: _State, action: Action, sketch: Sequence[Action]) -> _State | None:
    locations = state.location_map
    facts = set(state.facts)
    holding = state.holding
    name = action.operator
    args = action.arguments
    if name == "PICK":
        if holding is not None or args[0] not in locations:
            return None
        holding = args[0]
        locations.pop(args[0], None)
        facts = {fact for fact in facts if not fact.startswith(f"at({args[0]},")}
        facts.add(f"holding({args[0]})")
    elif name == "PLACE":
        if holding != args[0]:
            return None
        holding = None
        locations[args[0]] = args[1]
        facts.discard(f"holding({args[0]})")
        facts.add(f"at({args[0]},{args[1]})")
    elif name in {"POUR", "STIR"}:
        if holding != args[0] or args[1] not in locations:
            return None
        predicate = "poured" if name == "POUR" else "stirred"
        facts.add(f"{predicate}({args[0]},{args[1]})")
    elif name == "PLACE_SERVING_UTENSIL":
        if holding != args[0] or args[1] not in locations:
            return None
        holding = None
        locations[args[0]] = args[1]
        facts.discard(f"holding({args[0]})")
        facts.add(f"at({args[0]},{args[1]})")
        facts.add(f"served_with({args[1]},{args[0]})")
    elif name in {"OPEN", "INSPECT"}:
        if args[0] in state.opened:
            return None
        if holding is not None:
            return None
    elif name == "INSERT":
        if holding != args[0] or args[1] not in locations:
            return None
        holding = None
        facts.discard(f"holding({args[0]})")
        facts.add(f"inserted({args[0]},{args[1]})")
    elif name == "FASTEN":
        if holding != args[0] or f"inserted({args[1]},{args[2]})" not in facts:
            return None
        facts.add(f"fastened({args[0]},{args[1]},{args[2]})")
    else:
        return None
    opened = state.opened | ({args[0]} if name in {"OPEN", "INSPECT"} else set())
    if name in {"OPEN", "INSPECT"}:
        facts.add(f"open({args[0]})")
    executed = state.executed
    if executed < len(sketch) and action == sketch[executed]:
        executed += 1
    return _State(
        holding,
        tuple(sorted(locations.items())),
        frozenset(opened),
        frozenset(facts),
        executed,
    )


_LITERAL = re.compile(r"([a-z_]+)\(([^()]*)\)", re.IGNORECASE)
_PREDICATE_ARITY = {
    "at": 2,
    "holding": 1,
    "open": 1,
    "poured": 2,
    "stirred": 2,
    "served_with": 2,
    # The Kitchen goal contract states its placements as `placed(object,
    # region)`, and the model emits exactly that.  Without it here its own
    # required goal effect was rejected as an unsupported literal.
    "placed": 2,
    "inserted": 2,
    "fastened": 3,
}


def _goal_facts(
    literals: Sequence[str],
    observation: Observation,
    *,
    drop_unobserved: bool = False,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Normalized goal facts, plus the literals that were dropped.

    `drop_unobserved` decides what a literal naming an ID the robot cannot yet
    see means.  Under a single-shot protocol it is an error: the planner gets
    one look at the world, so a goal it cannot ground is a goal it cannot
    pursue, and silently discarding it would turn an unsatisfiable request into
    a plan that ignores half of what was asked.

    Under a multi-cycle protocol it is not an error but an unavoidable
    consequence of partial observability.  The Workshop goal is about a
    fastener and a driver that are shut inside storage at cycle one; no
    admissible ID for them exists until an inspection reveals them.  A model
    asked for goal literals there can only answer with names it does not have
    -- in practice the schema's own placeholders, `fastened(tool,fastener,
    target)` -- and rejecting the cycle for it stops the episode before it can
    perform the very inspection that would make the goal expressible.  So the
    literal is dropped for this cycle only, and recorded; the next cycle, which
    observes the revealed objects, can state it properly.
    """
    known = observation.object_ids | observation.region_ids
    result = set()
    dropped: list[str] = []
    for literal in literals:
        match = _LITERAL.fullmatch(literal.replace(" ", ""))
        if match is None:
            raise ValueError(f"unsupported goal literal syntax {literal!r}")
        predicate = match.group(1).lower()
        arguments = tuple(part for part in match.group(2).split(",") if part)
        # The goal contract names a placement `placed(object, destination)`;
        # the transition model records the same state as `at(object,
        # destination)`, which is what PLACE and PLACE_SERVING_UTENSIL add.
        # Accepting the contract's spelling and normalizing it here keeps the
        # goal expressible without giving the operators a second effect that
        # means the same thing.
        if predicate == "placed":
            predicate = "at"
        if predicate not in _PREDICATE_ARITY or len(arguments) != _PREDICATE_ARITY[predicate]:
            raise ValueError(f"unsupported goal literal {literal!r}")
        if any(argument not in known for argument in arguments):
            if not drop_unobserved:
                raise ValueError(f"goal literal references an unobserved ID: {literal!r}")
            dropped.append(literal)
            continue
        result.add(f"{predicate}({','.join(arguments)})")
    return frozenset(result), tuple(dropped)


def constrained_breadth_first_search(
    observation: Observation,
    grounded_actions: Sequence[Action],
    sketch: Sequence[Action],
    goal_literals: Sequence[str] = (),
    *,
    # Already-normalized goal facts, when the caller has resolved the literals
    # itself (the replanning protocol does, so it can record what it dropped).
    # Takes precedence over `goal_literals`.
    goal_facts: frozenset[str] | None = None,
    # Shape-admissibility test applied to *gap-filling* actions only -- the
    # ones this search invents to connect the model's sketch, never the sketch
    # actions themselves.  See `search_then_sample`'s `prune_gap_actions` for
    # why this exists and why the sketch is exempt.
    admissible: Callable[[Action], bool] | None = None,
    # Bounded so a sketch that has genuinely stalled fails in seconds rather
    # than grinding.  With the sketch-first rule above, an executable sketch
    # never approaches this; only the fall-back branching does, and there the
    # extra expansions buy almost nothing because the branching factor is the
    # whole grounded set.
    max_expansions: int = 20_000,
) -> tuple[Action, ...] | None:
    """Find a plan containing the VLM sketch as a subsequence.

    Ordered by how much of the sketch a state has executed, then by plan
    length.  Breadth-first over the whole grounded set cannot reach the depth
    a Kitchen sketch needs: with 365 grounded actions and a 14-action sketch,
    the 100,000-expansion budget is spent at sketch progress 3 of 14, so every
    Kitchen episode returned "no plan satisfies the Executed(i) subsequence
    constraints" for want of search, not for want of a plan.  Expanding
    sketch-advancing states first makes the common case -- a sketch that is
    already executable, which is what the model is asked for -- close to
    linear, while still returning a plan that contains the sketch as a
    subsequence and satisfies the goal.

    The trade is that the returned plan is no longer guaranteed shortest.
    OWL-TAMP needs a skeleton to hand to continuous sampling, not a minimal
    one, and the budget is retained so a genuinely unreachable goal still
    terminates.
    """
    initial = _initial_state(observation)
    goals = (
        goal_facts
        if goal_facts is not None
        else _goal_facts(goal_literals, observation)[0]
    )
    counter = itertools.count()
    fillers = (
        list(grounded_actions)
        if admissible is None
        else [action for action in grounded_actions if admissible(action)]
    )
    queue = [(-initial.executed, 0, next(counter), initial, ())]
    visited = {initial}
    expansions = 0
    while queue and expansions < max_expansions:
        _, depth, _, state, path = heapq.heappop(queue)
        expansions += 1
        if state.executed == len(sketch) and goals <= state.facts:
            return path
        # When the next sketch action is applicable, take it and nothing else.
        # The model is asked for a directly executable sequence, so in the
        # common case this walks the sketch in a straight line instead of
        # branching over all 365 grounded actions at every step.  Branching is
        # kept for states where the sketch has stalled, which is where a
        # gap-filling action is actually needed.
        if state.executed < len(sketch) and _transition(
            state, sketch[state.executed], sketch
        ) is not None:
            successors = [sketch[state.executed]]
        else:
            successors = fillers
        for action in successors:
            successor = _transition(state, action, sketch)
            if successor is None or successor in visited:
                continue
            visited.add(successor)
            heapq.heappush(queue, (
                -successor.executed, depth + 1, next(counter),
                successor, path + (action,),
            ))
    return None


def _replay(
    observation: Observation,
    skeleton: Sequence[Action],
    sketch_actions: Sequence[Action],
) -> list[_State] | None:
    """States before each skeleton action, or None if the skeleton is invalid.

    `states[i]` is the state in which `skeleton[i]` is applied, so the list is
    one longer than the skeleton.
    """
    state = _initial_state(observation)
    states = [state]
    for action in skeleton:
        successor = _transition(state, action, sketch_actions)
        if successor is None:
            return None
        state = successor
        states.append(state)
    return states


# PLACE and PLACE_SERVING_UTENSIL are this benchmark's "detach": the operators
# that release a held object onto a surface.
_DETACH_OPERATORS = frozenset({"PLACE", "PLACE_SERVING_UTENSIL"})


def appendix_plan_modification(
    observation: Observation,
    skeleton: Sequence[Action],
    sketch_actions: Sequence[Action],
    failed_index: int,
    rng: random.Random,
) -> tuple[Action, ...] | None:
    """OWL-TAMP Appendix A.1.1 task-plan modification, or None if inapplicable.

    The appendix describes what happens when "the sampling budget is exhausted
    for the first time and a new task plan is required":

        we employ a set of manually-engineered strategies to modify the task
        plan based on the most-recent failed operator (e.g. if the most
        recent-failed operator is a detach that was attempting to place an
        object atop a particular surface, and there are other objects atop that
        surface already, we randomly append a attach detach sequence to move
        one of those objects to a different part of the table).

    That single worked example is the only strategy the paper specifies, so it
    is the only one implemented here; inventing further strategies would be
    inventing method, not reproducing it.

    No VLM request is issued.  The appendix is explicit that the system "cannot
    recover from errors in the generated constraints themselves", so the
    constraints generated for this cycle are carried over unchanged and a
    modified skeleton is re-sampled against them.

    The relocation is inserted at the last point before the failure where the
    gripper is free, rather than immediately before the failed detach: at the
    detach itself the robot is by definition still holding the object it is
    trying to place, so a PICK of the blocking object is not applicable there.
    """
    failed = skeleton[failed_index]
    if failed.operator not in _DETACH_OPERATORS or len(failed.arguments) < 2:
        return None
    states = _replay(observation, skeleton, sketch_actions)
    if states is None:
        return None
    insert_at = failed_index
    while insert_at > 0 and states[insert_at].holding is not None:
        insert_at -= 1
    if states[insert_at].holding is not None:
        return None
    placed, destination = failed.arguments[0], failed.arguments[1]
    occupants = states[insert_at].location_map
    blocking = sorted(
        object_id
        for object_id, region in occupants.items()
        if region == destination and object_id != placed
    )
    elsewhere = sorted(observation.region_ids - {destination})
    if not blocking or not elsewhere:
        return None
    mover = rng.choice(blocking)
    relocation = (Action("PICK", (mover,)), Action("PLACE", (mover, rng.choice(elsewhere))))
    candidate = tuple(skeleton[:insert_at]) + relocation + tuple(skeleton[insert_at:])
    if _replay(observation, candidate, sketch_actions) is None:
        return None
    return candidate


def _sample_skeleton(
    skeleton: Sequence[Action],
    sketch_actions: Sequence[Action],
    by_action: Mapping[int, Sequence[Constraint]],
    oracle: SampleOracle,
    max_samples_per_action: int,
) -> tuple[int, int]:
    """Sample every action of one skeleton.

    Returns the number of samples drawn and the index of the first action whose
    sampling budget was exhausted, or -1 when the whole skeleton was accepted.
    """
    samples = 0
    sketch_cursor = 0
    for index, action in enumerate(skeleton):
        sketch_index = -1
        if sketch_cursor < len(sketch_actions) and action == sketch_actions[sketch_cursor]:
            sketch_index = sketch_cursor
            sketch_cursor += 1
        accepted = False
        for trial in range(max_samples_per_action):
            samples += 1
            if oracle(action, by_action.get(sketch_index, ()), trial):
                accepted = True
                break
        if not accepted:
            return samples, index
    return samples, -1


def search_then_sample(
    observation: Observation,
    grounded_actions: Sequence[Action],
    sketch: PlanSketch,
    constraints: Sequence[Constraint],
    oracle: SampleOracle,
    *,
    max_samples_per_action: int = PAPER_MAX_SAMPLES_PER_ACTION,
    max_skeletons: int = PAPER_MAX_SKELETONS,
    backtrack: bool = False,
    rng: random.Random | None = None,
    drop_unobserved_goals: bool = False,
    prune_gap_actions: bool = False,
) -> PlanningResult:
    """Run OWL-TAMP's symbolic-search then continuous-sampling protocol.

    With `backtrack` false -- the default, and what the single-shot `native`
    protocol uses -- exactly one skeleton is refined.  This benchmark's
    discrete operators are deterministic, so the Executed(i) search returns a
    single skeleton and the five-skeleton budget is never reached; the budget
    is retained for provenance and `skeletons_tested` is reported as 1.

    With `backtrack` true the appendix's recovery is enabled: a skeleton whose
    continuous sampling is exhausted is modified by
    `appendix_plan_modification` and re-refined, up to `max_skeletons`
    skeletons in total.  The default is off so that the already-reported
    single-shot grid keeps the semantics it was produced under; the flag must
    be requested explicitly.

    `prune_gap_actions` fixes a defect in the search, not in the method.
    Relaxed grounding admits every object as a PLACE destination, and the
    symbolic transition model does not check destinations either, so the search
    is free to invent a gap-filling step like `PLACE(manual_screwdriver,
    power_screwdriver)` purely to free the gripper.  No continuous value can
    ever satisfy that -- the scene's destinations are its regions plus the
    frame joint -- so the skeleton is dead on arrival and the whole 500-sample
    budget is spent proving it.  Observed directly: a Workshop cycle produced a
    13-action skeleton with three such placements and exhausted sampling at
    index 7, wasting the ten model calls that cycle had cost.

    With the flag set, an action the oracle cannot admit at all is not offered
    to the search as *filler*.  The model's own sketch actions are never
    filtered: if the sketch says `FASTEN(driver, screw, screwdriver)` with the
    wrong target, that is the method's error and it must fail on its merits.
    This only stops the search from inventing steps the refiner could never
    accept.

    The flag assumes the oracle is a deterministic shape test, which is true of
    the Workshop scene it is used for; it is off by default for that reason as
    well as to leave the reported grid untouched.  These admissibility calls
    are search-time, not refinement draws, and are deliberately not counted in
    `samples_tested`.
    """
    if sketch.status == "NO_PLAN":
        return PlanningResult("NO_PLAN", sketch, (), (), 0, 0, "model returned NO_PLAN")
    try:
        goals, dropped = _goal_facts(
            sketch.goal_literals, observation, drop_unobserved=drop_unobserved_goals
        )
        skeleton = constrained_breadth_first_search(
            observation, grounded_actions, sketch.actions, goal_facts=goals,
            admissible=(
                (lambda action: oracle(action, (), 0)) if prune_gap_actions else None
            ),
        )
    except ValueError as error:
        return PlanningResult(
            "INVALID_GOAL_LITERALS", sketch, (), tuple(constraints), 0, 0, str(error)
        )
    if skeleton is None:
        return PlanningResult(
            "NO_SYMBOLIC_PLAN", sketch, (), tuple(constraints), 1, 0,
            "no plan satisfies the Executed(i) subsequence constraints",
            dropped_goal_literals=dropped,
        )
    by_action: dict[int, list[Constraint]] = {
        index: [] for index in range(len(sketch.actions))
    }
    for constraint in constraints:
        by_action.setdefault(constraint.action_index, []).append(constraint)
    if rng is None:
        rng = random.Random(0)

    samples = 0
    skeletons_tested = 0
    failure = ""
    while skeletons_tested < max(1, max_skeletons):
        skeletons_tested += 1
        drawn, failed_index = _sample_skeleton(
            skeleton, sketch.actions, by_action, oracle, max_samples_per_action
        )
        samples += drawn
        if failed_index < 0:
            return PlanningResult(
                "PLAN", sketch, skeleton, tuple(constraints), skeletons_tested, samples,
                dropped_goal_literals=dropped,
            )
        failure = (
            f"continuous sampling exhausted for {skeleton[failed_index].operator}"
        )
        if not backtrack or skeletons_tested >= max(1, max_skeletons):
            break
        modified = appendix_plan_modification(
            observation, skeleton, sketch.actions, failed_index, rng
        )
        if modified is None:
            break
        skeleton = modified
    return PlanningResult(
        "NO_CONTINUOUS_PLAN", sketch, skeleton, tuple(constraints),
        skeletons_tested, samples, failure, dropped_goal_literals=dropped,
    )

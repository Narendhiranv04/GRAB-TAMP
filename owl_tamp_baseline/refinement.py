"""Executed(i) symbolic search followed by bounded continuous sampling."""

from __future__ import annotations

import heapq
import itertools

from collections import deque
from dataclasses import dataclass
import re
from typing import Protocol, Sequence

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


def _goal_facts(literals: Sequence[str], observation: Observation) -> frozenset[str]:
    known = observation.object_ids | observation.region_ids
    result = set()
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
            raise ValueError(f"goal literal references an unobserved ID: {literal!r}")
        result.add(f"{predicate}({','.join(arguments)})")
    return frozenset(result)


def constrained_breadth_first_search(
    observation: Observation,
    grounded_actions: Sequence[Action],
    sketch: Sequence[Action],
    goal_literals: Sequence[str] = (),
    *,
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
    goals = _goal_facts(goal_literals, observation)
    counter = itertools.count()
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
            successors = list(grounded_actions)
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


def search_then_sample(
    observation: Observation,
    grounded_actions: Sequence[Action],
    sketch: PlanSketch,
    constraints: Sequence[Constraint],
    oracle: SampleOracle,
    *,
    max_samples_per_action: int = PAPER_MAX_SAMPLES_PER_ACTION,
    max_skeletons: int = PAPER_MAX_SKELETONS,
) -> PlanningResult:
    """Run OWL-TAMP's symbolic-search then continuous-sampling protocol.

    This benchmark has deterministic discrete operators, so a single shortest
    skeleton is produced by the Executed(i) search.  The public five-skeleton
    budget is retained for provenance and future domains with skeleton choices.
    """
    if sketch.status == "NO_PLAN":
        return PlanningResult("NO_PLAN", sketch, (), (), 0, 0, "model returned NO_PLAN")
    try:
        skeleton = constrained_breadth_first_search(
            observation, grounded_actions, sketch.actions, sketch.goal_literals
        )
    except ValueError as error:
        return PlanningResult(
            "INVALID_GOAL_LITERALS", sketch, (), tuple(constraints), 0, 0, str(error)
        )
    if skeleton is None:
        return PlanningResult(
            "NO_SYMBOLIC_PLAN", sketch, (), tuple(constraints), 1, 0,
            "no plan satisfies the Executed(i) subsequence constraints",
        )
    samples = 0
    by_action = {index: [] for index in range(len(sketch.actions))}
    for constraint in constraints:
        by_action.setdefault(constraint.action_index, []).append(constraint)
    sketch_cursor = 0
    for action in skeleton:
        sketch_index = -1
        if (
            sketch_cursor < len(sketch.actions)
            and action == sketch.actions[sketch_cursor]
        ):
            sketch_index = sketch_cursor
            sketch_cursor += 1
        accepted = False
        for trial in range(max_samples_per_action):
            samples += 1
            if oracle(action, by_action.get(sketch_index, ()), trial):
                accepted = True
                break
        if not accepted:
            return PlanningResult(
                "NO_CONTINUOUS_PLAN", sketch, skeleton, tuple(constraints),
                min(1, max_skeletons), samples,
                f"continuous sampling exhausted for {action.operator}",
            )
    return PlanningResult(
        "PLAN", sketch, skeleton, tuple(constraints), min(1, max_skeletons), samples
    )

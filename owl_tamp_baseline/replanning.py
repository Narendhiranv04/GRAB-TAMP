"""OWL-TAMP with a replanning harness, bounded by a total model-call budget.

What the paper actually specifies
---------------------------------
OWL-TAMP's published recovery is entirely solver-side and failure-triggered.
Appendix A.1.1 backtracks over plan skeletons when "the sampling budget is
exhausted for the first time and a new task plan is required", modifying the
task plan with manually-engineered strategies keyed to the most-recent failed
operator, and Section 6 bounds that at five skeletons.  The VLM is never
re-queried: the paper states outright that the implementation "cannot recover
from errors in the generated constraints themselves", and its hardware section
executes "plans open-loop on hardware".

So the appendix mechanism -- implemented faithfully in
`refinement.appendix_plan_modification`, and enabled here -- only fires when
continuous sampling fails.  In this benchmark's Workshop it never fires: every
single-shot episode that plans at all produces the same three-action
INSPECT/INSPECT/INSPECT plan and executes it without a single failed action.
The method scores zero there not because anything fails but because all storage
starts closed, so at plan time there is no fastener or driver the model can
name, and a single-shot planner never gets a second look.

What this harness adds, and why it is named separately
------------------------------------------------------
This condition keeps the paper's per-cycle planner intact and keeps its
open-loop execution of a plan, and adds exactly one thing the paper does not
have: when a plan has been executed and the goal is not satisfied, the world is
observed afresh and a new cycle is planned.  That is an extension beyond
OWL-TAMP as published and must be reported as one.  It is not the paper's
real-robot condition, which is open-loop.

Two properties are deliberate, and keep the comparison honest:

* No failure-feedback prompt.  A new cycle receives only the fresh observable
  state and the original goal -- never the previous plan, the failed action, a
  failure code, or any privileged state.  Feedback-conditioned reprompting is
  VLM-TAMP's mechanism; borrowing it here would stop measuring OWL-TAMP.
* One total model-call budget across the whole episode, not a per-cycle one, so
  this column is comparable with the replan-budgeted baselines rather than
  being handed an unbounded number of requests.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from baseline_common.models import ActionResult, Observation

from .models import Action, Constraint, PlanningResult


class Planner(Protocol):
    response_trace: list[dict[str, Any]]
    trace: Mapping[str, Any]

    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        oracle: Callable[[Action, Sequence[Constraint], int], bool],
        *,
        movable_object_ids: Sequence[str] | None = None,
        max_vlm_requests: int | None = None,
    ) -> PlanningResult: ...


Observe = Callable[[], tuple[Observation, Sequence[Mapping[str, str]]]]
Execute = Callable[[Action], ActionResult]
GoalVerifier = Callable[[Observation], bool]
Oracle = Callable[[Action, Sequence[Constraint], int], bool]
MovableObjects = Callable[[Observation], Sequence[str] | None]


@dataclass(frozen=True)
class ReplanningResult:
    success: bool
    status: str
    planning_cycles: int
    replans: int
    model_calls: int
    max_model_calls: int
    executed_actions: int
    action_history: tuple[Mapping[str, Any], ...]
    planning_trace: tuple[Mapping[str, Any], ...]
    failure: str = ""
    # Cycles that produced no executable plan at all.
    no_plan_cycles: int = 0
    # Cycles whose plan executed without a single failed action.  Under the
    # paper's open-loop execution this is the common case, and it is the
    # counter that separates "the plan broke" from "the plan ran and was not
    # enough", which is the whole of the Workshop result.
    clean_cycles: int = 0
    # Skeletons refined across all cycles.  Anything above `planning_cycles`
    # means Appendix A.1.1's backtracking actually fired somewhere; equality
    # means it never did.
    skeletons_tested: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "planning_cycles": self.planning_cycles,
            "replans": self.replans,
            "model_calls": self.model_calls,
            "max_model_calls": self.max_model_calls,
            "executed_actions": self.executed_actions,
            "action_history": [dict(row) for row in self.action_history],
            "planning_trace": [dict(row) for row in self.planning_trace],
            "no_plan_cycles": self.no_plan_cycles,
            "clean_cycles": self.clean_cycles,
            "skeletons_tested": self.skeletons_tested,
            "failure": self.failure,
        }


class OWLTAMPReplanning:
    """Observe, plan a full OWL-TAMP cycle, execute it open-loop, repeat.

    A replan is every planning cycle after the first.  The sketch request and
    the per-action constraint requests inside one cycle are not replans; they
    are what one native OWL-TAMP planning call costs.
    """

    def __init__(
        self,
        planner: Planner,
        observe: Observe,
        execute: Execute,
        goal_verifier: GoalVerifier,
        oracle: Oracle,
        *,
        movable_objects: MovableObjects | None = None,
        # The binding budget.  Matched to the replan budget the feedback-driven
        # baselines are given so the columns cost the same.
        max_model_calls: int = 15,
        # A safety net only: the model-call budget is what should stop a run.
        max_cycles: int = 16,
        max_total_actions: int = 48,
        # A cycle that executes nothing leaves the world exactly as it was, so
        # the next cycle re-asks an identical question.  With a stochastic
        # decoder that is still a real retry -- it is the only retry OWL-TAMP
        # can make, since adding a failure-feedback prompt would replace its
        # mechanism with VLM-TAMP's -- so the default lets the model-call
        # budget be what stops the episode, exactly as the replan budget is
        # what stops the feedback-driven baselines.  The guard exists only to
        # bound a pathological cycle that consumes no budget at all.
        max_stagnant_cycles: int = 16,
    ):
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if max_cycles < 1 or max_total_actions < 1 or max_stagnant_cycles < 1:
            raise ValueError("cycle, action and stagnation budgets must be positive")
        self.planner = planner
        self.observe = observe
        self.execute = execute
        self.goal_verifier = goal_verifier
        self.oracle = oracle
        self.movable_objects = movable_objects or (lambda _observation: None)
        self.max_model_calls = max_model_calls
        self.max_cycles = max_cycles
        self.max_total_actions = max_total_actions
        self.max_stagnant_cycles = max_stagnant_cycles

    def run(self, goal: str) -> ReplanningResult:
        history: list[Mapping[str, Any]] = []
        traces: list[Mapping[str, Any]] = []
        model_calls = 0
        no_plan_cycles = 0
        clean_cycles = 0
        skeletons = 0
        stagnant = 0
        cycle = 0

        while cycle < self.max_cycles:
            observation, images = self.observe()
            if self.goal_verifier(observation):
                return self._result(
                    True, "GOAL_COMPLETE", cycle, model_calls, history, traces,
                    no_plan_cycles=no_plan_cycles, clean_cycles=clean_cycles,
                    skeletons_tested=skeletons,
                )
            remaining = self.max_model_calls - model_calls
            if remaining < 1:
                return self._result(
                    False, "MODEL_BUDGET_EXHAUSTED", cycle, model_calls, history,
                    traces,
                    f"The {self.max_model_calls}-model-call budget was exhausted.",
                    no_plan_cycles=no_plan_cycles, clean_cycles=clean_cycles,
                    skeletons_tested=skeletons,
                )

            cycle += 1
            result = self.planner.plan(
                goal,
                observation,
                images,
                self.oracle,
                movable_object_ids=self.movable_objects(observation),
                # The cycle may spend whatever is left of the episode budget:
                # one sketch request, then one constraint request per sketch
                # action until the budget runs out.  The planner records
                # `constraint_generation_complete` when it had to stop early.
                max_vlm_requests=remaining,
            )
            spent = len(self.planner.response_trace)
            model_calls += spent
            skeletons += result.skeletons_tested
            traces.append(
                {
                    "planning_cycle": cycle,
                    "observation_revision": observation.revision,
                    "model_calls_before": model_calls - spent,
                    "model_calls_spent": spent,
                    "result": result.as_dict(),
                    "model_trace": deepcopy(dict(self.planner.trace)),
                }
            )

            if result.status != "PLAN" or not result.actions:
                no_plan_cycles += 1
                stagnant += 1
                if stagnant >= self.max_stagnant_cycles:
                    return self._result(
                        False, "NO_PLAN", cycle, model_calls, history, traces,
                        result.failure or "OWL-TAMP returned no executable plan.",
                        no_plan_cycles=no_plan_cycles, clean_cycles=clean_cycles,
                        skeletons_tested=skeletons,
                    )
                # Nothing was applied, so the world is unchanged.  Re-observe
                # and re-plan while budget remains.
                continue

            # Open-loop execution of the whole plan, which is what the paper
            # does on hardware.  Execution stops at the first failed action;
            # the next cycle's only feedback is the world it then observes.
            executed_this_cycle = 0
            broke = False
            for action in result.actions:
                if len(history) >= self.max_total_actions:
                    return self._result(
                        False, "ACTION_BUDGET_EXHAUSTED", cycle, model_calls,
                        history, traces,
                        f"The {self.max_total_actions}-action budget was exhausted.",
                        no_plan_cycles=no_plan_cycles, clean_cycles=clean_cycles,
                        skeletons_tested=skeletons,
                    )
                outcome = self.execute(action)
                executed_this_cycle += 1
                history.append(
                    {
                        "planning_cycle": cycle,
                        "action": action.as_dict(),
                        "success": outcome.success,
                        "failure_code": outcome.failure_code,
                        "message": outcome.message,
                        "effects": list(outcome.effects),
                        "details": dict(outcome.details),
                    }
                )
                if not outcome.success:
                    broke = True
                    break
            if not broke:
                clean_cycles += 1
            stagnant = 0 if executed_this_cycle else stagnant + 1
            if stagnant >= self.max_stagnant_cycles:
                return self._result(
                    False, "NO_PROGRESS", cycle, model_calls, history, traces,
                    f"{stagnant} consecutive cycles changed nothing in the world.",
                    no_plan_cycles=no_plan_cycles, clean_cycles=clean_cycles,
                    skeletons_tested=skeletons,
                )

        # The cycle ceiling is a safety net; reaching it means the budget was
        # never the binding constraint, so say which limit actually stopped it.
        return self._result(
            False, "CYCLE_BUDGET_EXHAUSTED", cycle, model_calls, history, traces,
            f"The {self.max_cycles}-cycle ceiling was reached with "
            f"{self.max_model_calls - model_calls} model calls unspent.",
            no_plan_cycles=no_plan_cycles, clean_cycles=clean_cycles,
            skeletons_tested=skeletons,
        )

    def _result(
        self,
        success: bool,
        status: str,
        planning_cycles: int,
        model_calls: int,
        history: Sequence[Mapping[str, Any]],
        traces: Sequence[Mapping[str, Any]],
        failure: str = "",
        *,
        no_plan_cycles: int = 0,
        clean_cycles: int = 0,
        skeletons_tested: int = 0,
    ) -> ReplanningResult:
        return ReplanningResult(
            success=success,
            status=status,
            planning_cycles=planning_cycles,
            replans=max(0, planning_cycles - 1),
            model_calls=model_calls,
            max_model_calls=self.max_model_calls,
            executed_actions=len(history),
            action_history=tuple(history),
            planning_trace=tuple(traces),
            failure=failure,
            no_plan_cycles=no_plan_cycles,
            clean_cycles=clean_cycles,
            skeletons_tested=skeletons_tested,
        )

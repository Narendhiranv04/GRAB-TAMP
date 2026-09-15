"""Shadow layer for the region-inspection-order ablation.

Two things happen here, both entirely outside the pipeline:

1.  The region inspection order is switched between the deployed policy
    (``auto`` -- the FM's own ranking, system-completed) and a genuinely random
    shuffle of the whole region universe.  The random policy already exists in
    ``search_contract.freeze_search_region_contract``; this layer only chooses
    it and supplies a seed, so no pipeline behaviour is redefined here.

2.  Grounding and planning wall time are measured.  Neither is instrumented in
    the pipeline, and the ablation's whole point is that ordering moves
    inspection cost, so the two phases are timed separately from the FM call:

        grounding_seconds   region inspection + perception + phi
        planning_seconds    the symbolic planner

    The three domains do not share one solve path, so a single wrapper would
    silently measure only one of them.  Each is instrumented at its own entry:

        kitchen       domains.kitchen.run_to_plan       / plan_with_common_astar
        living_room   domains.living_room.run_to_plan   / run_living_room_symbolic_pipeline
        workshop      run.search_until_satisfied        / run.plan_with_common_astar

    For kitchen and living_room the planner runs *inside* the solve call, so its
    time is subtracted out; for workshop the two are siblings and nothing is
    subtracted.  The wrappers go on whichever namespace actually holds the name
    at call time: ``run.py`` binds the workshop pair at import, while the two
    ``run_to_plan`` functions are imported inside the dispatch body and so are
    patched on their defining modules.

The seed varies per trial.  A single fixed seed would give every one of the 320
trials the *same* shuffled order, which is a pattern, not randomness -- so the
seed is derived from the trial's own identity (output root, domain, variant),
making it distinct per trial and reproducible across reruns.

Living Room declares zero inspectable regions and the contract validator
rejects ``random`` (and any seed) for it.  It is therefore run under ``auto``
in both arms; with no regions to order, the arms are identical there by
construction, which the sidecar records explicitly rather than hiding.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

ORDER_MODES = ("auto", "random", "worst")

# Living Room has no inspectable regions; the contract validator refuses
# `random` and refuses a seed for it.
NO_SEARCH_DOMAINS = frozenset({"living_room"})

_SEED_SPACE = 2 ** 31


def trial_seed(seed_base: int, output_root: Any, domain: str, variant: str) -> int:
    """Deterministic per-trial seed.

    Keyed on the trial's own identity so that each of the 320 trials shuffles
    differently while a rerun of the same trial reproduces its order exactly.
    The output root carries the repeat/attempt directory, which is what makes
    repeats of one variant differ from each other.
    """
    key = f"{seed_base}|{Path(output_root).as_posix()}|{domain}|{variant}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_SPACE


def order_for_trial(order_mode: str, seed_base: int, output_root: Any,
                    domain: str, variant: str) -> tuple[str, int | None, str]:
    """Return (search_order, search_seed, applied_label) for one trial.

    The worst-case arm keeps the deployed source here and overrides the frozen
    contract's region list instead: its order is privileged, so it is not a
    policy the contract resolver is allowed to express.
    """
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unknown order mode {order_mode!r}; expected one of {ORDER_MODES}")
    if order_mode == "worst" and domain not in NO_SEARCH_DOMAINS:
        return "auto", None, "worst (privileged adversarial order, applied to the frozen contract)"
    if domain in NO_SEARCH_DOMAINS:
        # Not a silent downgrade: the domain has no regions to order, so both
        # arms are the same run and the sidecar says so.
        return "auto", None, f"auto (domain {domain} declares no inspectable regions)"
    if order_mode == "auto":
        return "auto", None, "auto"
    return "random", trial_seed(seed_base, output_root, domain, variant), "random"


class _PhaseClock:
    """Wall time for the solve phase and the planner nested inside it.

    ``solve`` is the whole grounding/search phase as each domain implements it.
    For kitchen and living_room the planner is called from inside that phase, so
    its time is recorded separately and subtracted; for workshop the planner is
    a sibling of the search call and nothing is subtracted. ``_depth`` is what
    tells the two cases apart, rather than a hard-coded per-domain rule.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.solve = 0.0
        self.planning = 0.0
        self.planning_nested = 0.0
        self.solve_calls = 0
        self.planning_calls = 0
        self._depth = 0

    @contextlib.contextmanager
    def solve_phase(self):
        self._depth += 1
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._depth -= 1
            # Only the outermost solve contributes, so a domain that nests solve
            # calls cannot double-count its own time.
            if self._depth == 0:
                self.solve += elapsed
                self.solve_calls += 1

    @contextlib.contextmanager
    def planning_phase(self):
        nested = self._depth > 0
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.planning += elapsed
            self.planning_calls += 1
            if nested:
                self.planning_nested += elapsed

    @property
    def grounding_seconds(self) -> float:
        # Never report a negative phase: clock skew or a planner that outlives
        # its solve frame would otherwise produce nonsense.
        return max(0.0, self.solve - self.planning_nested)


def _install(target: Any, name: str, wrap) -> tuple[Any, str, Any] | None:
    """Wrap ``target.name`` in place, returning what is needed to restore it.

    Returns None when the attribute is absent, so that a renamed entry point
    surfaces as a missing-instrumentation warning instead of an import crash.
    """
    original = getattr(target, name, None)
    if original is None:
        return None
    setattr(target, name, wrap(original))
    return (target, name, original)


def _install_worst_case_contract() -> list:
    """Rewrite the frozen contract's region order to the adversarial one.

    The worst-case order is oracle information, so it is imposed here rather
    than added as a policy the resolver could pick: nothing inside the pipeline
    gains a way to ask for it. Everything else about the contract -- its
    validation, its provenance trace, the no-search decision for Living Room --
    is whatever the real resolver produced.
    """
    import dataclasses

    from mujoco_scenes.functional_tamp_pipeline import run as run_module
    from mujoco_scenes.fm_worst_case_order import worst_case_order

    original = run_module.freeze_search_region_contract

    def patched(specification, domain=None, source="auto", **kwargs):
        contract = original(specification, domain, source, **kwargs)
        eff_domain = domain or getattr(specification, "domain", None)
        order = worst_case_order(str(eff_domain), str(kwargs.get("variant") or ""))
        if order is None or contract.no_search_required:
            return contract
        # Keep only the regions this contract actually declared, so the override
        # cannot widen the search beyond what the resolver authorised.
        declared = set(contract.canonical_region_ids)
        reordered = tuple(r for r in order if r in declared)
        if set(reordered) != declared:
            return contract
        return dataclasses.replace(
            contract,
            canonical_region_ids=reordered,
            source="PRIVILEGED_GT_WORST_CASE_DIAGNOSTIC",
        )

    run_module.freeze_search_region_contract = patched
    return [(run_module, "freeze_search_region_contract", original)]


def _instrument_phases(clock: _PhaseClock) -> tuple[list, list[str]]:
    """Install the solve/planner timers on every domain's own entry points."""
    from mujoco_scenes.functional_tamp_pipeline import run as run_module
    from mujoco_scenes.functional_tamp_pipeline.domains import kitchen as kitchen_module
    from mujoco_scenes.functional_tamp_pipeline.domains import living_room as living_room_module

    def as_solve(original):
        def wrapper(*args, **kwargs):
            with clock.solve_phase():
                return original(*args, **kwargs)
        return wrapper

    def as_planning(original):
        def wrapper(*args, **kwargs):
            with clock.planning_phase():
                return original(*args, **kwargs)
        return wrapper

    targets = [
        (kitchen_module, "run_to_plan", as_solve),
        (living_room_module, "run_to_plan", as_solve),
        (run_module, "search_until_satisfied", as_solve),
        (kitchen_module, "plan_with_common_astar", as_planning),
        (living_room_module, "run_living_room_symbolic_pipeline", as_planning),
        (run_module, "plan_with_common_astar", as_planning),
    ]
    restores, missing = [], []
    for module, name, wrap in targets:
        handle = _install(module, name, wrap)
        if handle is None:
            missing.append(f"{module.__name__}.{name}")
        else:
            restores.append(handle)
    return restores, missing


@contextlib.contextmanager
def search_order_shadow(evaluator_module: Any, *, order_mode: str, seed_base: int = 0):
    """Install the ordering choice and the phase timers for one evaluation pass.

    ``evaluator_module`` is the already-imported evaluator; it binds
    ``run_pipeline`` into its own namespace, so the wrapper is installed there.
    """
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unknown order mode {order_mode!r}; expected one of {ORDER_MODES}")

    clock = _PhaseClock()
    stats: dict[str, Any] = {
        "trials": 0, "random_trials": 0, "auto_trials": 0, "worst_trials": 0,
        "untimed_trials": 0, "missing_instrumentation": [], "sidecars": [],
    }

    original_run_pipeline = evaluator_module.run_pipeline
    restores, missing = _instrument_phases(clock)
    stats["missing_instrumentation"] = missing
    if order_mode == "worst":
        restores.extend(_install_worst_case_contract())

    def patched_run_pipeline(*args, **kwargs):
        if args:
            raise TypeError("search_order_shadow requires run_pipeline to be called with keyword arguments")
        domain = kwargs["domain"]
        variant = kwargs["variant"]
        mode = kwargs.get("mode", "vlm")
        output_root = kwargs["output_root"]

        order, seed, applied = order_for_trial(order_mode, seed_base, output_root, domain, variant)
        kwargs["search_order"] = order
        kwargs["search_seed"] = seed

        clock.reset()
        started = time.perf_counter()
        try:
            return original_run_pipeline(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            run_dir = Path(output_root) / domain / variant / mode
            sidecar = {
                "domain": domain,
                "variant": variant,
                "mode": mode,
                "order_mode_requested": order_mode,
                "order_mode_applied": applied,
                "search_order": order,
                "search_seed": seed,
                "seed_base": seed_base,
                "grounding_seconds": round(clock.grounding_seconds, 6),
                "planning_seconds": round(clock.planning, 6),
                "solve_seconds": round(clock.solve, 6),
                "planning_nested_seconds": round(clock.planning_nested, 6),
                "grounding_calls": clock.solve_calls,
                "planning_calls": clock.planning_calls,
                "trial_seconds": round(elapsed, 6),
                "run_dir": str(run_dir),
            }
            stats["trials"] += 1
            if order == "random":
                key = "random_trials"
            elif order_mode == "worst" and domain not in NO_SEARCH_DOMAINS:
                key = "worst_trials"
            else:
                key = "auto_trials"
            stats[key] += 1
            # A trial that never reached the solve phase (an FM spec failure) has
            # no grounding time to report; that is a real outcome, not a missing
            # measurement, so it is counted rather than imputed.
            if clock.solve_calls == 0:
                stats["untimed_trials"] += 1
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "shadow_search_order_timing.json").write_text(
                    json.dumps(sidecar, indent=2), encoding="utf-8"
                )
            except Exception as error:  # a lost sidecar must not fail the trial
                print(f"  [search-order-shadow] sidecar write failed for {run_dir}: {error}", flush=True)
            stats["sidecars"].append(sidecar)

    evaluator_module.run_pipeline = patched_run_pipeline
    try:
        yield stats
    finally:
        evaluator_module.run_pipeline = original_run_pipeline
        for module, name, original in restores:
            setattr(module, name, original)

"""Shadow layer: zero-shot fallback for the canonicalization vocabularies.

Auxiliary study, not part of the method. The pipeline is not modified; this
patches four resolver entry points in-process and restores them on exit.

It augments rather than replaces. The hand-written cue tables keep priority and
are consulted first; the classifier is asked only when they return nothing. A
trial the current system already handles therefore takes the identical path, so
the arm cannot regress what works today and any difference is attributable to
coverage that did not exist before.

    capability  robot_capability_registry.extract_operation_semantic_candidates
    relation    relation_interpreter.extract_relation_semantic_candidates
    region      kitchen / workshop region proposal resolvers
    role        semantic_compiler._map_role

Endpoint filtering, validation and every downstream stage are untouched: a
zero-shot suggestion still has to survive the same checks a cue match does.

Abstention is configurable and reported, never tuned against outcomes. The
default accepts the top-ranked candidate, because that is the setting whose
accuracy was measured offline (0.74-0.89 depending on layer) and so the setting
whose behaviour is known. Raising the floors trades coverage for caution: a
mis-mapped capability is worse than an unmapped one, since an unmapped operation
fails closed while a wrong one plans against the wrong preconditions. Both
settings are legitimate arms; neither may be selected after seeing the result.

Note the scores are softmax over two to seven labels, so they concentrate near
0.5 and a high floor silently disables the fallback entirely. Any threshold used
must be sanity-checked against the acceptance counts this layer reports.
"""
from __future__ import annotations

import contextlib
import functools
import os
import threading
from pathlib import Path
from typing import Any

# Declared up front; override only to run a deliberately stated arm.
MIN_SCORE = float(os.environ.get("TAMP_ZS_MIN_SCORE", "0.0"))
MIN_MARGIN = float(os.environ.get("TAMP_ZS_MIN_MARGIN", "0.0"))
MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
TEMPLATE = "This refers to {}."

_LOCK = threading.Lock()
_PIPE = None
_CACHE: dict[tuple, list[tuple[str, float]]] = {}
STATS: dict[str, int] = {}


def _classifier():
    global _PIPE
    with _LOCK:
        if _PIPE is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            import torch
            from transformers import pipeline
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
            _PIPE = pipeline("zero-shot-classification", model=MODEL, device=-1)
        return _PIPE


def _rank(premise: str, labels: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Rank canonical ids for one premise; labels are (id, hypothesis text)."""
    if not premise.strip() or not labels:
        return []
    if len(labels) == 1:
        return [(labels[0][0], 1.0)]
    key = (premise.lower(), tuple(i for i, _ in labels))
    if key in _CACHE:
        return _CACHE[key]
    texts = [t for _, t in labels]
    res = _classifier()(premise, texts, hypothesis_template=TEMPLATE, multi_label=False)
    back = {t: i for i, t in labels}
    ranked = [(back[l], float(s)) for l, s in zip(res["labels"], res["scores"])]
    _CACHE[key] = ranked
    return ranked


def _accept(ranked: list[tuple[str, float]]) -> str | None:
    if not ranked:
        return None
    top = ranked[0][1]
    margin = top - ranked[1][1] if len(ranked) > 1 else top
    if top >= MIN_SCORE and margin >= MIN_MARGIN:
        return ranked[0][0]
    return None


def _bump(name: str) -> None:
    STATS[name] = STATS.get(name, 0) + 1


@contextlib.contextmanager
def zs_canonicalization():
    """Install the zero-shot fallback on the four resolver entry points."""
    from mujoco_scenes.functional_tamp_pipeline import robot_capability_registry as RC
    from mujoco_scenes.functional_tamp_pipeline import relation_interpreter as RI
    from mujoco_scenes.functional_tamp_pipeline import predicate_registry as PR
    from mujoco_scenes import kitchen_vlm_functional_graph as KV
    from mujoco_scenes.workshop_phase1 import requirements as WR

    STATS.clear()
    restores: list[tuple[Any, str, Any]] = []

    # ---- capability ----
    orig_ops = RC.extract_operation_semantic_candidates

    @functools.wraps(orig_ops)
    def ops(domain, raw_phrase, participant_roles=()):
        found = orig_ops(domain, raw_phrase, participant_roles)
        if found:
            _bump("capability_cue")
            return found
        caps = list(RC.get_robot_capabilities(domain))
        if not caps:
            return found
        labels = [(c.capability_id, f"{c.capability_id}: {c.semantic_description}") for c in caps]
        parts = ", ".join(str(p) for p in participant_roles) or "unspecified participants"
        premise = f'an operation the plan calls "{raw_phrase}" involving {parts}'
        pick = _accept(_rank(premise, labels))
        if pick is None:
            _bump("capability_abstain")
            return found
        _bump("capability_zs")
        return tuple(c for c in caps if c.capability_id == pick)

    RC.extract_operation_semantic_candidates = ops
    restores.append((RC, "extract_operation_semantic_candidates", orig_ops))
    if getattr(RI, "extract_operation_semantic_candidates", None) is orig_ops:
        RI.extract_operation_semantic_candidates = ops
        restores.append((RI, "extract_operation_semantic_candidates", orig_ops))

    # ---- relation ----
    orig_rel = RI.extract_relation_semantic_candidates

    @functools.wraps(orig_rel)
    def rel(domain, raw_phrase):
        found = orig_rel(domain, raw_phrase)
        if found:
            _bump("relation_cue")
            return found
        sigs = [s for s in PR.get_active_predicates(domain) if int(getattr(s, "arity", 2) or 2) == 2]
        if not sigs:
            return found
        labels = [(s.name, f"{s.name}: {getattr(s, 'description', '') or s.name}") for s in sigs]
        pick = _accept(_rank(f'a requirement the plan states as "{raw_phrase}"', labels))
        if pick is None:
            _bump("relation_abstain")
            return found
        _bump("relation_zs")
        # Mirror the shape the caller expects from a cue match.
        return tuple(
            RI.RelationSemanticCandidate(
                predicate_name=s.name,
                direction="FORWARD",
                category=str(getattr(s, "category", "") or "PHYSICAL_VERIFIER"),
            )
            for s in sigs if s.name == pick
        )

    RI.extract_relation_semantic_candidates = rel
    restores.append((RI, "extract_relation_semantic_candidates", orig_rel))

    # ---- regions ----
    KITCHEN_REGION_TEXT = {
        "C1": "C1: a closed storage cabinet mounted high on the wall, on the left",
        "C2": "C2: a closed storage cabinet mounted high on the wall, on the right",
        "D1": "D1: a closed drawer underneath the table, on the left",
        "D2": "D2: a closed drawer underneath the table, on the right",
        "B1": "B1: a container resting on top of the table, on the right",
    }
    WORKSHOP_REGION_TEXT = {
        "LEFT_DRAWER": "LEFT_DRAWER: the left drawer",
        "RIGHT_DRAWER": "RIGHT_DRAWER: the right drawer",
        "TOOL_CABINET": "TOOL_CABINET: the tool cabinet",
    }

    def region_patch(module, attr, table, tag):
        orig = getattr(module, attr)

        @functools.wraps(orig)
        def resolver(proposal):
            try:
                found = orig(proposal)
            except Exception:
                found = None
            if found:
                _bump(f"{tag}_cue")
                return found
            if isinstance(proposal, dict):
                premise = f"{proposal.get('label','')}: {proposal.get('visual_description','')}".strip(": ")
            else:
                premise = str(proposal)
            pick = _accept(_rank(premise, list(table.items())))
            if pick is None:
                _bump(f"{tag}_abstain")
                return found
            _bump(f"{tag}_zs")
            return pick

        setattr(module, attr, resolver)
        restores.append((module, attr, orig))

    region_patch(KV, "resolve_kitchen_region_proposal", KITCHEN_REGION_TEXT, "region_kitchen")
    region_patch(WR, "resolve_workshop_region_proposal", WORKSHOP_REGION_TEXT, "region_workshop")

    # ---- role ----
    # Roles block more failing trials than any other single layer, and unlike the
    # other three the ontology publishes no prose for them -- only the categories
    # each role accepts. The label is composed from the canonical name and those
    # categories, which is mechanical; writing role descriptions by hand would be
    # the practice under test.
    from mujoco_scenes.functional_tamp_pipeline import semantic_compiler as SC
    from mujoco_scenes.functional_tamp_pipeline.system_context_registry import (
        get_domain_selectable_roles, get_domain_system_fixed_anchors,
    )
    import yaml

    _ont = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "mujoco_scenes/configs/runtime_functional_semantic_ontology.yaml").read_text(encoding="utf-8")
    )

    def role_labels(domain: str) -> list[tuple[str, str]]:
        block = (_ont.get("domains") or {}).get(domain) or {}
        roles = block.get("roles") or block
        allowed = set(get_domain_selectable_roles(domain)) | set(get_domain_system_fixed_anchors(domain))
        out = []
        for name, body in sorted(roles.items()):
            if allowed and name not in allowed:
                continue
            cats = (body or {}).get("accepted_categories") or []
            text = f"{name}: the {name.replace('_', ' ').lower()}"
            if cats:
                text += f", typically a {' or '.join(cats[:4])}"
            out.append((name, text))
        return out

    orig_map_role = SC._map_role

    @functools.wraps(orig_map_role)
    def map_role(domain, role, doc):
        try:
            mapped, rule = orig_map_role(domain, role, doc)
        except Exception:
            mapped, rule = None, "RESOLVER_RAISED"
        if mapped:
            _bump("role_cue")
            return mapped, rule
        labels = role_labels(str(domain))
        if not labels:
            return mapped, rule
        bits = [f'a role the plan calls "{role.get("id", "")}"']
        fn = str(role.get("function") or role.get("description") or "").strip().rstrip(".")
        if fn:
            bits.append(f"whose job is {fn[0].lower() + fn[1:]}" if fn else "")
        cats = [str(c) for c in (role.get("candidate_categories") or [])][:3]
        if cats:
            bits.append(f"and which could be a {' or '.join(cats)}")
        props = [str(c) for c in (role.get("required_properties") or [])][:3]
        if props:
            bits.append(f"[must be: {', '.join(props)}]")
        pick = _accept(_rank(" ".join(b for b in bits if b), labels))
        if pick is None:
            _bump("role_abstain")
            return mapped, rule
        _bump("role_zs")
        return pick, "ZERO_SHOT_SEMANTIC_FALLBACK"

    SC._map_role = map_role
    restores.append((SC, "_map_role", orig_map_role))

    try:
        yield STATS
    finally:
        for module, attr, orig in restores:
            setattr(module, attr, orig)

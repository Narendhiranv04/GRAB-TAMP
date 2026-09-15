"""Privileged worst-case inspection order, for the ablation's upper bound only.

This is oracle information and is not part of the method. Choosing an order that
delays the regions actually holding things requires knowing what each region
holds, which the pipeline is never told. It exists to bound the ablation from
the other side: if a deliberately adversarial order costs no more than the FM's
ranking, the benchmark cannot discriminate inspection-order policies at all, and
the null result between FM and random says something about the benchmark rather
than about the ranking.

The order is by ascending stored-object count, empty regions first, ties broken
by canonical order. Empty-first is the adversarial part -- a region holding
nothing can never satisfy anything, so opening it is pure waste. Ordering the
rest by ascending count generalizes that to workshop, where almost every region
holds something and an empty-only rule would not discriminate.

This is a lower bound on the true worst case rather than the exact maximum: the
exact worst order would delay whichever region holds the objects this particular
task needs, which depends on the task, not just on occupancy. Counts come from
KitchenScene.config.container_contents and
WorkshopScene.privileged_get_storage_contents, both ground-truth accessors.

Living Room declares no inspectable regions, so it has no worst case and is
absent here; it runs under the deployed policy in every arm.
"""
from __future__ import annotations

WORST_CASE_ORDERS = {
    ("kitchen", "K1"): ('D1', 'D2', 'C2', 'B1', 'C1'),  # contents (0, 0, 0, 0, 0)
    ("kitchen", "K10"): ('D2', 'D1', 'C2', 'B1', 'C1'),  # contents (0, 1, 1, 1, 1)
    ("kitchen", "K11"): ('D1', 'D2', 'C2', 'B1', 'C1'),  # contents (1, 1, 1, 1, 1)
    ("kitchen", "K12"): ('D1', 'D2', 'C2', 'B1', 'C1'),  # contents (1, 1, 1, 1, 1)
    ("kitchen", "K2"): ('D1', 'D2', 'B1', 'C1', 'C2'),  # contents (0, 0, 0, 0, 1)
    ("kitchen", "K3"): ('D1', 'D2', 'C2', 'C1', 'B1'),  # contents (0, 0, 0, 0, 1)
    ("kitchen", "K4"): ('D1', 'D2', 'C1', 'C2', 'B1'),  # contents (0, 0, 0, 1, 1)
    ("kitchen", "K5"): ('C2', 'B1', 'C1', 'D1', 'D2'),  # contents (0, 0, 0, 1, 1)
    ("kitchen", "K6"): ('D1', 'D2', 'C2', 'B1', 'C1'),  # contents (1, 1, 1, 1, 1)
    ("kitchen", "K7"): ('D1', 'D2', 'C2', 'C1', 'B1'),  # contents (0, 0, 0, 0, 1)
    ("kitchen", "K8"): ('D1', 'D2', 'B1', 'C1', 'C2'),  # contents (0, 0, 0, 0, 1)
    ("kitchen", "K9"): ('C1', 'D1', 'D2', 'C2', 'B1'),  # contents (0, 1, 1, 1, 1)
    ("workshop", "W1"): ('RIGHT_DRAWER', 'TOOL_CABINET', 'LEFT_DRAWER'),  # contents (1, 1, 2)
    ("workshop", "W10"): ('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET'),  # contents (1, 1, 1)
    ("workshop", "W2"): ('RIGHT_DRAWER', 'TOOL_CABINET', 'LEFT_DRAWER'),  # contents (1, 1, 2)
    ("workshop", "W3"): ('RIGHT_DRAWER', 'TOOL_CABINET', 'LEFT_DRAWER'),  # contents (1, 1, 2)
    ("workshop", "W4"): ('RIGHT_DRAWER', 'TOOL_CABINET', 'LEFT_DRAWER'),  # contents (1, 1, 2)
    ("workshop", "W5"): ('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET'),  # contents (1, 1, 2)
    ("workshop", "W6"): ('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET'),  # contents (1, 1, 2)
    ("workshop", "W7"): ('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET'),  # contents (1, 1, 1)
    ("workshop", "W8"): ('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET'),  # contents (1, 1, 1)
    ("workshop", "W9"): ('TOOL_CABINET', 'LEFT_DRAWER', 'RIGHT_DRAWER'),  # contents (0, 1, 1)
}


def worst_case_order(domain: str, variant: str) -> tuple[str, ...] | None:
    """Adversarial order for one variant, or None when there is none defined."""
    return WORST_CASE_ORDERS.get((domain, variant))

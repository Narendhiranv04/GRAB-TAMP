"""Function phrases for the retrieval baseline.

Deliberately free of category nouns.  Prompting with "side table" or "cup"
would supply the answer and reduce the baseline to a label lookup, which is
what the proposed method's semantic evidence already does.  These phrases
describe only what the role must *do* or how it must be shaped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    key: str
    phrase: str
    count: int
    kind: str  # "object" or "region"


LIVING_ROOM_ROLES = (
    Role(
        key="personal_support",
        phrase="a flat raised surface within arm's reach of a single seated person",
        count=2,
        kind="region",
    ),
    Role(
        key="shared_support",
        phrase="a flat raised surface reachable from two seats at once",
        count=1,
        kind="region",
    ),
    Role(
        key="drink_vessel",
        phrase="a small open vessel for drinking from",
        count=2,
        kind="object",
    ),
    Role(
        key="under_dish",
        phrase="a small shallow flat dish that another object rests on",
        count=2,
        kind="object",
    ),
    Role(
        key="handheld_control",
        phrase="a small handheld device covered in buttons",
        count=1,
        kind="object",
    ),
)

LIVING_ROOM_TASK_TEMPLATE = (
    # (vessel index, dish index, personal support index)
    (0, 0, 0),
    (1, 1, 1),
)


# Workshop: find a compatible screw and driver in the closed storage regions,
# insert the screw into the workbench repair hole and drive it home, then leave
# the driver on the workbench.  As above, no category nouns: "screwdriver" or
# "screw" would hand the baseline its answer and turn CLIP retrieval into a
# label lookup, which is precisely the capability under test.
WORKSHOP_ROLES = (
    Role(
        key="turning_tool",
        phrase="a hand tool with a long shaft and a shaped tip for turning",
        count=1,
        kind="object",
    ),
    Role(
        key="threaded_part",
        phrase="a small metal part with a spiral ridge along its body",
        count=1,
        kind="object",
    ),
)

# Storage is closed at the start, so retrieval cannot score what it cannot see.
# The template opens regions in the scene's fixed inspection order and only
# then scores the revealed candidates -- the same discovery discipline the
# other baselines follow.
WORKSHOP_INSPECTION_FIRST = True


# Kitchen: brew one coffee per person using a water source and a grounds
# source, stir each, then serve every vessel and pair each soup vessel with its
# own eating utensil.  Same discipline as above -- no category nouns.  Note
# that two pairs of roles are deliberately close in wording, because they are
# close in appearance: the stirring implement and the eating utensil are both
# slender implements, and the two vessel kinds are both open vessels.  That is
# the distinction functional grounding is supposed to make and similarity
# retrieval is not.
KITCHEN_ROLES = (
    Role(
        key="water_source",
        phrase="a closed vessel with a handle and a spout for pouring hot liquid",
        count=1,
        kind="object",
    ),
    Role(
        key="grounds_source",
        phrase="a closed container holding a dry granular substance",
        count=1,
        kind="object",
    ),
    Role(
        key="stir_tool",
        phrase="a slender rigid implement for agitating a liquid",
        count=1,
        kind="object",
    ),
    Role(
        key="drink_vessel",
        phrase="a small deep open vessel for drinking a hot liquid from",
        count=2,
        kind="object",
    ),
    Role(
        key="food_vessel",
        phrase="a wide shallow open vessel for holding liquid food",
        count=2,
        kind="object",
    ),
    Role(
        key="eating_utensil",
        phrase="a slender implement with a shallow scoop at one end",
        count=2,
        kind="object",
    ),
)

# Ordered task skeleton, derived from the published ground-truth sequence
# (K1: PICK x9, POUR x4, PLACE x7, STIR x2, PLACE_SERVING_UTENSIL x2).
#
# Each source is picked once, applied to every drink vessel, and returned to
# the countertop before the next is picked, because the robot has one gripper.
# Serving interleaves: a food vessel is placed and then immediately paired with
# its utensil, which is what the ground truth does and what the serving-slot
# allocator expects.
KITCHEN_TASK_SKELETON = (
    ("USE_SOURCE", "water_source"),
    ("USE_SOURCE", "grounds_source"),
    ("STIR_ALL", "stir_tool"),
    ("SERVE_ALL", "drink_vessel"),
    ("SERVE_PAIRED", "food_vessel"),
)

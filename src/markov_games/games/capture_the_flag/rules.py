"""The Capture the Flag transition function.

This module is pure: no RNG, no I/O, no mutation of its arguments, no knowledge
of the environment wrapper. ``transition`` computes the successor state from the
*full joint action at once* — never by walking the agents in order and mutating
as it goes. Agent order appears exactly once, as a documented tie-break for
respawn placement, and it cannot change anyone's outcome.

The rules implemented here are specified in ``docs/capture-the-flag.md`` §3-§7.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

from .config import AGENT_TEAMS, NUM_AGENTS, CtfConfig, Position, other_team
from .state import CtfState


class Action(IntEnum):
    """The five moves available to every agent, every step."""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


#: Cell offset of each action. ``UP`` decreases ``y``: the origin is top-left.
OFFSETS: tuple[Position, ...] = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))

NUM_ACTIONS = len(Action)

JointAction = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransitionEvents:
    """What happened during one timestep, for tests, rendering and ``info``.

    Attributes:
        blocked: agents whose move was cancelled by a collision rule.
        tagged: agents that were tagged and respawned.
        pickups: agents that picked up an enemy flag.
        captures: teams that scored.
    """

    blocked: tuple[int, ...] = ()
    tagged: tuple[int, ...] = ()
    pickups: tuple[int, ...] = ()
    captures: tuple[int, ...] = ()


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Step 1-2: movement (docs/capture-the-flag.md §4)
# ---------------------------------------------------------------------------


def resolve_movement(
    config: CtfConfig, positions: tuple[Position, ...], actions: JointAction
) -> tuple[tuple[Position, ...], tuple[int, ...]]:
    """Resolve a simultaneous joint move.

    Each agent proposes a target; conflicting proposals are cancelled until a
    fixpoint is reached. Cancellation is monotone — a blocked agent never becomes
    unblocked — so this terminates, and the result is independent of the order in
    which agents are examined.

    Returns the new positions and the indices of the agents that were blocked.
    """
    intents: list[Position] = []
    for pos, action in zip(positions, actions, strict=True):
        dx, dy = OFFSETS[action]
        target = (pos[0] + dx, pos[1] + dy)
        # Walking into a wall is a no-op, not an error.
        intents.append(target if config.in_bounds(target) else pos)

    blocked = [False] * len(positions)
    while True:
        targets = [positions[i] if blocked[i] else intents[i] for i in range(len(positions))]
        changed = False

        # R1 - swap: two agents exchanging cells both stay.
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if blocked[i] or blocked[j]:
                    continue
                if targets[i] == positions[j] and targets[j] == positions[i]:
                    blocked[i] = blocked[j] = True
                    changed = True

        # R2 - contested cell: if two or more agents want it, none of them get it.
        # A stationary agent is a claimant on its own cell, so this also covers
        # "you cannot walk into an occupied cell", and cascades down chains.
        if not changed:
            claims = Counter(targets)
            for i in range(len(positions)):
                if not blocked[i] and claims[targets[i]] > 1:
                    blocked[i] = True
                    changed = True

        if not changed:
            return tuple(targets), tuple(i for i, b in enumerate(blocked) if b)


# ---------------------------------------------------------------------------
# Step 3-5: tagging, flag drops, respawns (docs/capture-the-flag.md §5)
# ---------------------------------------------------------------------------


def find_tagged(config: CtfConfig, positions: tuple[Position, ...]) -> tuple[int, ...]:
    """Agents standing in enemy territory with an opponent within ``tag_radius``.

    Evaluated simultaneously for every agent, so a mutual tag across the midline
    is possible and symmetric.
    """
    tagged = []
    for i, pos in enumerate(positions):
        if not config.is_enemy_territory(pos, AGENT_TEAMS[i]):
            continue  # safe in the neutral midline and at home
        if any(
            AGENT_TEAMS[j] != AGENT_TEAMS[i] and manhattan(pos, positions[j]) <= config.tag_radius
            for j in range(len(positions))
        ):
            tagged.append(i)
    return tuple(tagged)


def respawn(
    config: CtfConfig, positions: tuple[Position, ...], tagged: tuple[int, ...]
) -> tuple[Position, ...]:
    """Return tagged agents to their spawns, preserving one-agent-per-cell.

    A spawn that is occupied falls back to the nearest free cell, ordered by
    Manhattan distance and then by ``(y, x)``. Simultaneously tagged agents are
    placed in global agent order; that ordering decides placement only, and
    cannot change who was tagged or what they were carrying.
    """
    if not tagged:
        return positions

    occupied = {positions[i] for i in range(len(positions)) if i not in tagged}
    placed = list(positions)
    for agent in tagged:
        spawn = config.spawns[agent]
        if spawn not in occupied:
            cell = spawn
        else:
            cell = min(
                (c for c in config.cells() if c not in occupied),
                key=lambda c: (manhattan(c, spawn), c[1], c[0]),
            )
        placed[agent] = cell
        occupied.add(cell)
    return tuple(placed)


# ---------------------------------------------------------------------------
# The whole step
# ---------------------------------------------------------------------------


def transition(
    config: CtfConfig, state: CtfState, actions: JointAction
) -> tuple[CtfState, TransitionEvents]:
    """Advance the game by one timestep.

    The successor is a pure function of ``(config, state, actions)``; ``state`` is
    not modified. The seven steps below are exactly those of
    ``docs/capture-the-flag.md`` §4-§5, applied to the whole joint action.
    """
    if len(actions) != NUM_AGENTS:
        raise ValueError(f"expected {NUM_AGENTS} actions, got {len(actions)}")
    if any(not 0 <= a < NUM_ACTIONS for a in actions):
        raise ValueError(f"actions must be in [0, {NUM_ACTIONS}), got {actions}")

    # 1-2: everyone moves at once.
    positions, blocked = resolve_movement(config, state.positions, actions)

    # 3: everyone is tagged at once, judged on the post-move positions.
    tagged = find_tagged(config, positions)
    carried_by = list(state.carried_by)

    # 4: a tagged carrier sends the flag straight home. No loose flags.
    for agent in tagged:
        for team, carrier in enumerate(carried_by):
            if carrier == agent:
                carried_by[team] = None

    # 5: tagged agents go back to their spawns.
    positions = respawn(config, positions, tagged)

    # 6: pick up the enemy flag by standing on the enemy base. Cells hold one
    # agent, so a pickup is never contested.
    pickups = []
    for agent, pos in enumerate(positions):
        if agent in tagged:
            continue
        stolen = other_team(AGENT_TEAMS[agent])
        if carried_by[stolen] is None and pos == config.bases[stolen]:
            carried_by[stolen] = agent
            pickups.append(agent)

    # 7: carry it to your own base to score. The captured flag goes home; the
    # agents stay where they are.
    scores = list(state.scores)
    captures = []
    for team, carrier in enumerate(carried_by):
        if carrier is None:
            continue
        scorer = AGENT_TEAMS[carrier]
        if positions[carrier] == config.bases[scorer]:
            scores[scorer] += 1
            carried_by[team] = None
            captures.append(scorer)

    successor = CtfState(
        positions=positions,
        carried_by=(carried_by[0], carried_by[1]),
        scores=(scores[0], scores[1]),
        step=state.step + 1,
    )
    events = TransitionEvents(
        blocked=blocked,
        tagged=tagged,
        pickups=tuple(pickups),
        captures=tuple(captures),
    )
    return successor, events


def team_rewards(events: TransitionEvents) -> tuple[float, float]:
    """Sparse, shared, zero-sum: +1 to a scoring team, -1 to the other (§6)."""
    blue = events.captures.count(0)
    red = events.captures.count(1)
    return (float(blue - red), float(red - blue))

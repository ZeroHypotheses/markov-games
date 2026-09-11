"""Movement resolution: docs/capture-the-flag.md §4.

Every rule in that section gets a test that pins the exact successor positions.
"""

from __future__ import annotations

import pytest

from markov_games.games.capture_the_flag.config import CtfConfig
from markov_games.games.capture_the_flag.rules import Action, resolve_movement

STAY, UP, DOWN, LEFT, RIGHT = Action


def move(config: CtfConfig, positions, actions):
    return resolve_movement(config, tuple(positions), tuple(int(a) for a in actions))


def test_agents_move_independently_when_nothing_conflicts(config):
    positions, blocked = move(config, [(1, 1), (1, 3), (5, 1), (5, 3)], [RIGHT, UP, LEFT, DOWN])
    assert positions == ((2, 1), (1, 2), (4, 1), (5, 4))
    assert blocked == ()


def test_walking_into_a_wall_is_a_no_op(config):
    positions, blocked = move(config, [(0, 0), (1, 3), (6, 4), (5, 1)], [LEFT, STAY, DOWN, STAY])
    assert positions[0] == (0, 0)
    assert positions[2] == (6, 4)
    assert blocked == ()  # a wall is not a collision, it is simply no movement


def test_a_swap_is_cancelled_for_both_agents(config):
    positions, blocked = move(config, [(1, 2), (2, 2), (5, 1), (5, 3)], [RIGHT, LEFT, STAY, STAY])
    assert positions[:2] == ((1, 2), (2, 2))
    assert blocked == (0, 1)


def test_a_contested_cell_is_won_by_nobody(config):
    positions, blocked = move(config, [(1, 1), (1, 3), (5, 1), (5, 3)], [DOWN, UP, STAY, STAY])
    assert positions[:2] == ((1, 1), (1, 3))
    assert blocked == (0, 1)


def test_cannot_move_into_a_stationary_agent(config):
    positions, blocked = move(config, [(1, 2), (2, 2), (5, 1), (5, 3)], [RIGHT, STAY, STAY, STAY])
    assert positions[0] == (1, 2)
    assert 0 in blocked


def test_a_blocked_chain_cascades_backwards(config):
    # blue_0 -> blue_1 -> red_0, and red_0 is standing still: all three stay.
    positions, blocked = move(config, [(3, 2), (4, 2), (5, 2), (5, 4)], [RIGHT, RIGHT, STAY, STAY])
    assert positions == ((3, 2), (4, 2), (5, 2), (5, 4))
    assert set(blocked) >= {0, 1}


def test_a_rotation_through_occupied_cells_succeeds(config):
    # Four agents on a 2x2 block, each stepping into the next one's cell.
    positions, blocked = move(config, [(2, 1), (3, 1), (3, 2), (2, 2)], [RIGHT, DOWN, LEFT, UP])
    assert positions == ((3, 1), (3, 2), (2, 2), (2, 1))
    assert blocked == ()


def test_resolution_does_not_depend_on_agent_order(config):
    """The same geometry with the two contenders in swapped slots resolves alike."""
    first, _ = move(config, [(1, 1), (1, 3), (5, 1), (5, 3)], [DOWN, UP, STAY, STAY])
    second, _ = move(config, [(1, 3), (1, 1), (5, 1), (5, 3)], [UP, DOWN, STAY, STAY])
    assert first[:2] == ((1, 1), (1, 3))
    assert second[:2] == ((1, 3), (1, 1))


@pytest.mark.parametrize("width,height", [(4, 3), (5, 5), (7, 5), (9, 7)])
def test_agents_never_share_a_cell(width, height):
    """The one-agent-per-cell invariant, exhaustively over every joint action."""
    import itertools

    config = CtfConfig(width=width, height=height)
    positions = config.spawns
    for actions in itertools.product(range(5), repeat=4):
        result, _ = move(config, positions, actions)
        assert len(set(result)) == 4
        assert all(config.in_bounds(cell) for cell in result)

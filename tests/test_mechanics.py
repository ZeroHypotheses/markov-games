"""Tagging, flags, scoring, termination: docs/capture-the-flag.md §5-§7."""

from __future__ import annotations

from conftest import make_state

from markov_games.games.capture_the_flag.config import BLUE, RED, CtfConfig
from markov_games.games.capture_the_flag.rules import Action, team_rewards, transition
from markov_games.games.capture_the_flag.state import initial_state

STAY, UP, DOWN, LEFT, RIGHT = Action
ALL_STAY = (STAY, STAY, STAY, STAY)


# -- tagging ---------------------------------------------------------------


def test_an_intruder_adjacent_to_a_defender_is_tagged(config):
    state = make_state(((4, 2), (0, 0), (5, 2), (6, 4)))
    successor, events = transition(config, state, ALL_STAY)
    assert events.tagged == (0,)
    assert successor.positions[0] == config.spawns[0]
    assert successor.positions[2] == (5, 2)  # the defender does not move


def test_an_agent_in_its_own_half_is_never_tagged(config):
    state = make_state(((1, 2), (0, 0), (2, 2), (6, 4)))
    successor, events = transition(config, state, ALL_STAY)
    assert events.tagged == (2,)  # the red agent is the one out of position
    assert successor.positions[0] == (1, 2)


def test_the_neutral_midline_is_safe(config):
    state = make_state(((3, 2), (0, 0), (4, 2), (6, 4)))
    _, events = transition(config, state, ALL_STAY)
    assert events.tagged == ()


def test_tagging_is_simultaneous_and_can_be_mutual():
    # An even-width board has no neutral column, so the halves touch.
    config = CtfConfig(width=6, height=5)
    state = make_state(((3, 2), (0, 0), (2, 2), (5, 4)))
    _, events = transition(config, state, ALL_STAY)
    assert events.tagged == (0, 2)


def test_a_tagged_agent_respawns_on_the_nearest_free_cell(config):
    # blue_1 is sitting on blue_0's spawn, so blue_0 has to go somewhere else.
    state = make_state(((4, 2), (1, 1), (5, 2), (6, 4)))
    successor, events = transition(config, state, ALL_STAY)
    assert events.tagged == (0,)
    assert successor.positions[0] == (1, 0)  # nearest free to (1,1), ordered by (dist, y, x)
    assert len(set(successor.positions)) == 4


def test_tag_radius_is_configurable():
    config = CtfConfig(tag_radius=2)
    state = make_state(((4, 2), (0, 0), (6, 2), (6, 4)))
    _, events = transition(config, state, ALL_STAY)
    assert events.tagged == (0,)


# -- flags -----------------------------------------------------------------


def test_standing_on_the_enemy_base_picks_up_the_flag(config):
    state = make_state(((5, 2), (0, 0), (4, 0), (4, 4)))
    successor, events = transition(config, state, (RIGHT, STAY, STAY, STAY))
    assert events.pickups == (0,)
    assert successor.carried_by[RED] == 0
    assert successor.flag_position(config, RED) == (6, 2)


def test_a_tagged_agent_does_not_pick_up_on_the_same_step(config):
    state = make_state(((5, 2), (0, 0), (6, 1), (4, 4)))
    successor, events = transition(config, state, (RIGHT, STAY, STAY, STAY))
    assert events.tagged == (0,)
    assert events.pickups == ()
    assert successor.carried_by[RED] is None


def test_tagging_a_carrier_sends_the_flag_home(config):
    state = make_state(((4, 2), (0, 0), (5, 2), (6, 4)), carried_by=(None, 0))
    successor, events = transition(config, state, ALL_STAY)
    assert events.tagged == (0,)
    assert successor.carried_by[RED] is None
    assert successor.flag_position(config, RED) == config.bases[RED]


def test_a_flag_already_carried_cannot_be_picked_up_again(config):
    state = make_state(((1, 2), (6, 2), (0, 0), (0, 4)), carried_by=(None, 0))
    successor, events = transition(config, state, ALL_STAY)
    assert events.pickups == ()
    assert successor.carried_by[RED] == 0


# -- scoring, termination, truncation --------------------------------------


def test_carrying_the_enemy_flag_home_scores(config):
    state = make_state(((1, 2), (0, 0), (5, 1), (5, 3)), carried_by=(None, 0))
    successor, events = transition(config, state, (LEFT, STAY, STAY, STAY))
    assert events.captures == (BLUE,)
    assert successor.scores == (1, 0)
    assert successor.carried_by[RED] is None  # the flag goes back to red's base
    assert successor.positions[0] == (0, 2)  # the world is not reset


def test_a_capture_is_plus_one_to_the_team_and_minus_one_to_the_other(config):
    state = make_state(((1, 2), (0, 0), (5, 1), (5, 3)), carried_by=(None, 0))
    _, events = transition(config, state, (LEFT, STAY, STAY, STAY))
    assert team_rewards(events) == (1.0, -1.0)


def test_nothing_happening_is_worth_nothing(config):
    _, events = transition(config, initial_state(config), ALL_STAY)
    assert team_rewards(events) == (0.0, 0.0)


def test_reaching_the_score_limit_wins(config):
    state = make_state(((1, 2), (0, 0), (5, 1), (5, 3)), carried_by=(None, 0), scores=(2, 0))
    successor, _ = transition(config, state, (LEFT, STAY, STAY, STAY))
    assert successor.scores == (3, 0)
    assert successor.winner(config) == BLUE


def test_nobody_has_won_before_the_limit(config):
    assert initial_state(config).winner(config) is None


# -- purity ----------------------------------------------------------------


def test_the_transition_does_not_mutate_its_input(config):
    state = make_state(((4, 2), (0, 0), (5, 2), (6, 4)), carried_by=(None, 0))
    before = (state.positions, state.carried_by, state.scores, state.step)
    transition(config, state, (RIGHT, UP, LEFT, STAY))
    assert (state.positions, state.carried_by, state.scores, state.step) == before


def test_the_same_input_always_gives_the_same_successor(config):
    state = make_state(((2, 2), (1, 1), (4, 2), (5, 3)))
    actions = (RIGHT, DOWN, LEFT, UP)
    first, _ = transition(config, state, actions)
    second, _ = transition(config, state, actions)
    assert first == second


def test_states_are_hashable_so_a_tabular_baseline_can_key_on_them(config):
    assert len({initial_state(config), initial_state(config)}) == 1


def test_an_invalid_action_is_rejected(config):
    import pytest

    with pytest.raises(ValueError):
        transition(config, initial_state(config), (0, 0, 0, 99))
    with pytest.raises(ValueError):
        transition(config, initial_state(config), (0, 0, 0))

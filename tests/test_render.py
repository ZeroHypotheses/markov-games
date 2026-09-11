"""Rendering. ``rgb_array`` must work with no display attached — CI renders it."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_state

from markov_games import CaptureTheFlagEnv, CtfConfig
from markov_games.games.capture_the_flag.render import CtfRenderer
from markov_games.games.capture_the_flag.state import initial_state


def expected_shape(config: CtfConfig, cell: int) -> tuple[int, int, int]:
    margin = max(8, cell // 6)
    hud = max(56, cell)
    return (config.height * cell + hud + 2 * margin, config.width * cell + 2 * margin, 3)


def test_rgb_array_has_the_right_shape_and_dtype():
    config = CtfConfig()
    environment = CaptureTheFlagEnv(config=config, render_mode="rgb_array", cell_size=48)
    environment.reset(seed=0)
    frame = environment.render()
    assert frame.shape == expected_shape(config, 48)
    assert frame.dtype == np.uint8
    environment.close()


def test_rendering_is_a_pure_function_of_the_state():
    config = CtfConfig()
    renderer = CtfRenderer(config, mode="rgb_array", cell_size=32)
    state = initial_state(config)
    first = renderer.render(state)
    second = renderer.render(state)
    assert np.array_equal(first, second)
    renderer.close()


def test_the_frame_changes_when_the_game_does():
    config = CtfConfig()
    renderer = CtfRenderer(config, mode="rgb_array", cell_size=32)
    start = renderer.render(initial_state(config))
    moved = renderer.render(make_state(((2, 1), (1, 3), (5, 1), (5, 3))))
    carrying = renderer.render(
        make_state(((2, 1), (1, 3), (5, 1), (5, 3)), carried_by=(None, 0), scores=(1, 2))
    )
    assert not np.array_equal(start, moved)
    assert not np.array_equal(moved, carrying)
    renderer.close()


@pytest.mark.parametrize("width,height,cell", [(4, 3, 24), (5, 5, 40), (9, 7, 16)])
def test_any_board_size_renders(width, height, cell):
    config = CtfConfig(width=width, height=height)
    renderer = CtfRenderer(config, mode="rgb_array", cell_size=cell)
    frame = renderer.render(initial_state(config))
    assert frame.shape == expected_shape(config, cell)
    renderer.close()


def test_no_render_mode_means_no_frame():
    environment = CaptureTheFlagEnv()
    environment.reset(seed=0)
    assert environment.render() is None
    environment.close()


def test_an_unknown_renderer_mode_is_rejected():
    with pytest.raises(ValueError):
        CtfRenderer(CtfConfig(), mode="ansi")


def test_a_rollout_produces_one_frame_per_step_plus_the_initial_one():
    from markov_games.policies import RandomPolicy
    from markov_games.rollout import run_episode

    config = CtfConfig(max_cycles=12)
    environment = CaptureTheFlagEnv(config=config, render_mode="rgb_array", cell_size=24)
    result = run_episode(
        environment, RandomPolicy(environment.possible_agents, seed=0), seed=0, collect_frames=True
    )
    assert len(result.frames) == result.steps + 1
    assert all(frame.shape == expected_shape(config, 24) for frame in result.frames)
    environment.close()

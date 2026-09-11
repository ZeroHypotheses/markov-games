"""PettingZoo Parallel API conformance, seeding and determinism."""

from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from markov_games import CaptureTheFlagEnv, CtfConfig
from markov_games.games.capture_the_flag.config import AGENT_NAMES
from markov_games.games.capture_the_flag.observations import OBSERVATION_SIZE, STATE_SIZE
from markov_games.policies import RandomPolicy


@pytest.fixture
def env() -> CaptureTheFlagEnv:
    environment = CaptureTheFlagEnv()
    yield environment
    environment.close()


def test_conforms_to_the_parallel_api():
    environment = CaptureTheFlagEnv(config=CtfConfig(max_cycles=32))
    parallel_api_test(environment, num_cycles=200)
    environment.close()


def test_the_agents_are_the_four_we_promised(env):
    assert env.possible_agents == list(AGENT_NAMES) == ["blue_0", "blue_1", "red_0", "red_1"]


def test_reset_returns_one_observation_per_agent(env):
    observations, infos = env.reset(seed=0)
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    for agent, observation in observations.items():
        assert observation.shape == (OBSERVATION_SIZE,)
        assert observation.dtype == np.float32
        assert env.observation_space(agent).contains(observation)


def test_spaces_are_stable_objects(env):
    for agent in env.possible_agents:
        assert env.observation_space(agent) is env.observation_space(agent)
        assert env.action_space(agent) is env.action_space(agent)
        assert env.action_space(agent).n == 5


def test_the_global_state_is_separate_from_the_observations(env):
    env.reset(seed=0)
    assert env.state().shape == (STATE_SIZE,)
    assert env.state_space.contains(env.state())


def test_rewards_are_zero_sum(env):
    observations, _ = env.reset(seed=3)
    policy = RandomPolicy(env.possible_agents, seed=3)
    while env.agents:
        observations, rewards, _, _, _ = env.step(policy(observations))
        assert sum(rewards.values()) == pytest.approx(0.0)
        assert rewards["blue_0"] == rewards["blue_1"]
        assert rewards["red_0"] == rewards["red_1"]


def test_the_episode_ends_and_the_agent_list_empties(env):
    observations, _ = env.reset(seed=0)
    policy = RandomPolicy(env.possible_agents, seed=0)
    steps = 0
    while env.agents:
        observations, _, terminations, truncations, _ = env.step(policy(observations))
        steps += 1
        assert steps <= env.config.max_cycles
    assert env.agents == []
    assert any(terminations.values()) or any(truncations.values())


def test_an_episode_without_a_winner_is_truncated_not_terminated():
    environment = CaptureTheFlagEnv(config=CtfConfig(max_cycles=8))
    environment.reset(seed=0)
    for _ in range(8):
        _, _, terminations, truncations, _ = environment.step(dict.fromkeys(environment.agents, 0))
    assert all(truncations.values())
    assert not any(terminations.values())
    environment.close()


def test_missing_actions_are_an_error(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step({"blue_0": 0})


def test_the_same_seed_replays_the_same_episode():
    """Same seed plus same policy seed must give a byte-identical trajectory."""

    def rollout(seed: int) -> list[tuple]:
        environment = CaptureTheFlagEnv()
        observations, _ = environment.reset(seed=seed)
        policy = RandomPolicy(environment.possible_agents, seed=seed)
        trace = []
        while environment.agents:
            actions = policy(observations)
            observations, _, _, _, _ = environment.step(actions)
            state = environment.game_state
            trace.append((tuple(actions.values()), state.positions, state.carried_by, state.scores))
        environment.close()
        return trace

    assert rollout(11) == rollout(11)
    assert rollout(11) != rollout(12)


def test_action_space_sampling_is_seeded():
    first = CaptureTheFlagEnv()
    second = CaptureTheFlagEnv()
    first.reset(seed=5)
    second.reset(seed=5)
    assert [first.action_space(a).sample() for a in first.possible_agents] == [
        second.action_space(a).sample() for a in second.possible_agents
    ]
    first.close()
    second.close()


@pytest.mark.parametrize("width,height", [(4, 3), (5, 5), (9, 7)])
def test_the_board_size_is_configurable(width, height):
    environment = CaptureTheFlagEnv(config=CtfConfig(width=width, height=height, max_cycles=16))
    observations, _ = environment.reset(seed=1)
    assert observations["blue_0"].shape == (OBSERVATION_SIZE,)
    policy = RandomPolicy(environment.possible_agents, seed=1)
    while environment.agents:
        observations, _, _, _, _ = environment.step(policy(observations))
        assert len(set(environment.game_state.positions)) == 4
    environment.close()


@pytest.mark.parametrize("kwargs", [{"width": 3}, {"height": 2}, {"score_to_win": 0}])
def test_an_impossible_board_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CtfConfig(**kwargs)


def test_an_unknown_render_mode_is_rejected():
    with pytest.raises(ValueError):
        CaptureTheFlagEnv(render_mode="ascii")

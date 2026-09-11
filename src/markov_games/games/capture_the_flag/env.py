"""PettingZoo ``ParallelEnv`` wrapper around the Capture the Flag rules.

The environment is bookkeeping only: it owns the current state, the agent list,
the spaces and the renderer. All of the game logic lives in :mod:`rules`, which
knows nothing about this class.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from . import rules
from .config import AGENT_NAMES, AGENT_TEAMS, CtfConfig
from .observations import (
    OBSERVATION_SIZE,
    STATE_SIZE,
    encode_global_state,
    encode_observation,
)
from .render import CtfRenderer
from .rules import NUM_ACTIONS, Action
from .state import CtfState, initial_state

__all__ = ["Action", "CaptureTheFlagEnv"]


class CaptureTheFlagEnv(ParallelEnv):
    """2v2 Capture the Flag on a small symmetric grid.

    Four agents — ``blue_0``, ``blue_1``, ``red_0``, ``red_1`` — act
    simultaneously. Rewards are sparse, shared within a team and zero-sum: +1 to
    a team that returns the enemy flag to its own base, -1 to the other team.

    The full rules are specified in ``docs/capture-the-flag.md``. The dynamics are
    deterministic; ``seed`` controls action-space sampling only.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "name": "capture_the_flag_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 6,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        config: CtfConfig | None = None,
        render_mode: str | None = None,
        cell_size: int = 72,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")

        self.config = config or CtfConfig()
        self.render_mode = render_mode
        self.cell_size = cell_size

        self.possible_agents: list[str] = list(AGENT_NAMES)
        self.agents: list[str] = []

        obs_space = spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self._observation_spaces = {a: obs_space for a in self.possible_agents}
        self._action_spaces = {a: spaces.Discrete(NUM_ACTIONS) for a in self.possible_agents}
        self.state_space = spaces.Box(-1.0, 1.0, shape=(STATE_SIZE,), dtype=np.float32)

        self._state: CtfState = initial_state(self.config)
        self._renderer: CtfRenderer | None = None
        self._last_events = rules.TransitionEvents()

    # -- spaces -----------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    # -- core loop --------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Start a new episode.

        The dynamics are deterministic, so ``seed`` only seeds the action spaces
        — that is what makes a random rollout reproducible.
        """
        if seed is not None:
            for offset, agent in enumerate(self.possible_agents):
                self._action_spaces[agent].seed(seed + offset)

        self.agents = list(self.possible_agents)
        self._state = initial_state(self.config)
        self._last_events = rules.TransitionEvents()

        if self.render_mode == "human":
            self.render()
        return self._observations(), {a: {} for a in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Apply one simultaneous joint action."""
        if not self.agents:
            return {}, {}, {}, {}, {}
        missing = set(self.agents) - set(actions)
        if missing:
            raise ValueError(f"missing actions for {sorted(missing)}")

        joint = tuple(int(actions[a]) for a in self.possible_agents)
        self._state, events = rules.transition(self.config, self._state, joint)
        self._last_events = events

        blue_reward, red_reward = rules.team_rewards(events)
        rewards = {
            agent: (blue_reward, red_reward)[AGENT_TEAMS[i]]
            for i, agent in enumerate(self.possible_agents)
        }

        terminated = self._state.winner(self.config) is not None
        truncated = not terminated and self._state.step >= self.config.max_cycles
        terminations = dict.fromkeys(self.agents, terminated)
        truncations = dict.fromkeys(self.agents, truncated)
        infos = {agent: self._info(i) for i, agent in enumerate(self.possible_agents)}

        observations = self._observations()
        if terminated or truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    # -- views on the state -----------------------------------------------

    def state(self) -> np.ndarray:
        """The global state vector, for centralised critics."""
        return encode_global_state(self.config, self._state)

    @property
    def game_state(self) -> CtfState:
        """The underlying :class:`CtfState`. Read-only by construction (frozen)."""
        return self._state

    def _observations(self) -> dict[str, np.ndarray]:
        return {
            agent: encode_observation(self.config, self._state, i)
            for i, agent in enumerate(self.possible_agents)
        }

    def _info(self, agent: int) -> dict[str, Any]:
        events = self._last_events
        return {
            "tagged": agent in events.tagged,
            "blocked": agent in events.blocked,
            "picked_up_flag": agent in events.pickups,
            "scores": self._state.scores,
        }

    # -- rendering --------------------------------------------------------

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = CtfRenderer(
                self.config,
                mode=self.render_mode,
                cell_size=self.cell_size,
                fps=int(self.metadata["render_fps"]),
            )
        return self._renderer.render(self._state)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

"""What each agent is told, as opposed to what the world knows.

Observations are near-complete today, but they are built here rather than read
off ``CtfState`` directly, so that introducing partial observability later is a
change to this file alone. See ``docs/capture-the-flag.md`` §8.

Two encodings live here:

* :func:`encode_observation` - **egocentric**: self, teammate, opponents, own
  flag, enemy flag. Both teams see a structurally identical problem, which is
  what lets a single network be shared across a team later.
* :func:`encode_global_state` - **absolute**: all four agents in global order.
  This is the extra information a centralised critic is allowed, and nothing
  else in the repo may read it.
"""

from __future__ import annotations

import numpy as np

from .config import AGENT_TEAMS, BLUE, NUM_AGENTS, RED, CtfConfig, Position, other_team
from .state import CtfState

#: Length of a per-agent observation vector.
OBSERVATION_SIZE = 21

#: Length of the global state vector.
STATE_SIZE = 21


def _normalised(config: CtfConfig, cell: Position) -> tuple[float, float]:
    return (
        cell[0] / max(config.width - 1, 1),
        cell[1] / max(config.height - 1, 1),
    )


def teammate_of(agent: int) -> int:
    """The other agent on ``agent``'s team. Teams are the adjacent pairs (0,1), (2,3)."""
    return agent ^ 1


def opponents_of(agent: int) -> tuple[int, ...]:
    """Both opponents, in global agent order."""
    return tuple(i for i in range(NUM_AGENTS) if AGENT_TEAMS[i] != AGENT_TEAMS[agent])


def _agent_block(config: CtfConfig, state: CtfState, agent: int) -> list[float]:
    x, y = _normalised(config, state.positions[agent])
    return [x, y, float(state.carrying(agent) is not None)]


def encode_observation(config: CtfConfig, state: CtfState, agent: int) -> np.ndarray:
    """The observation of one agent, ordered relative to itself."""
    team = AGENT_TEAMS[agent]
    enemy = other_team(team)

    values: list[float] = []
    values += _agent_block(config, state, agent)
    values += _agent_block(config, state, teammate_of(agent))
    for opponent in opponents_of(agent):
        values += _agent_block(config, state, opponent)

    own_flag_x, own_flag_y = _normalised(config, state.flag_position(config, team))
    values += [own_flag_x, own_flag_y, float(state.carried_by[team] is not None)]

    enemy_flag_x, enemy_flag_y = _normalised(config, state.flag_position(config, enemy))
    values += [enemy_flag_x, enemy_flag_y, float(state.carried_by[enemy] is not None)]

    values.append((state.scores[team] - state.scores[enemy]) / config.score_to_win)
    values.append(1.0 - state.step / config.max_cycles)
    values.append(float(team == BLUE))

    return np.asarray(values, dtype=np.float32)


def encode_global_state(config: CtfConfig, state: CtfState) -> np.ndarray:
    """The full state as a vector, in absolute (non-egocentric) ordering."""
    values: list[float] = []
    for agent in range(NUM_AGENTS):
        values += _agent_block(config, state, agent)
    for team in (BLUE, RED):
        flag_x, flag_y = _normalised(config, state.flag_position(config, team))
        values += [flag_x, flag_y, float(state.carried_by[team] is not None)]
    values += [state.scores[BLUE] / config.score_to_win, state.scores[RED] / config.score_to_win]
    values.append(1.0 - state.step / config.max_cycles)
    return np.asarray(values, dtype=np.float32)

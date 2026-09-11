"""The global state of a Capture the Flag game.

``CtfState`` is frozen and hashable, which makes it usable directly as a
dictionary key for a tabular baseline, and makes accidental in-place mutation
inside the transition function impossible. See ``docs/capture-the-flag.md`` §2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import AGENT_TEAMS, BLUE, RED, CtfConfig, Position, other_team


@dataclass(frozen=True, slots=True)
class CtfState:
    """Everything the world knows.

    Attributes:
        positions: one cell per agent, in global agent order. At most one agent
            occupies any cell.
        carried_by: for each team's flag, the index of the agent carrying it, or
            ``None`` if it is home on its base. Only opponents can carry a flag.
        scores: captures made by each team.
        step: elapsed timesteps since reset.
    """

    positions: tuple[Position, ...]
    carried_by: tuple[int | None, int | None]
    scores: tuple[int, int]
    step: int

    # -- derived ----------------------------------------------------------

    def flag_position(self, config: CtfConfig, team: int) -> Position:
        """Where ``team``'s flag is: its base, or whoever is carrying it."""
        carrier = self.carried_by[team]
        return config.bases[team] if carrier is None else self.positions[carrier]

    def carrying(self, agent: int) -> int | None:
        """The team whose flag ``agent`` is carrying, or ``None``."""
        stolen = other_team(AGENT_TEAMS[agent])
        return stolen if self.carried_by[stolen] == agent else None

    def winner(self, config: CtfConfig) -> int | None:
        """The team that has reached ``score_to_win``, if any."""
        for team in (BLUE, RED):
            if self.scores[team] >= config.score_to_win:
                return team
        return None

    def replace(self, **changes: object) -> CtfState:
        """A copy with fields replaced — states are never mutated in place."""
        return replace(self, **changes)  # type: ignore[arg-type]


def initial_state(config: CtfConfig) -> CtfState:
    """The opening position: agents on their spawns, both flags home, 0-0."""
    return CtfState(
        positions=config.spawns,
        carried_by=(None, None),
        scores=(0, 0),
        step=0,
    )

"""Geometry and limits for Capture the Flag.

Every tunable number lives here, typed and defaulted, so that a reader can see
the whole parameterisation of the game in one screen. See
``docs/capture-the-flag.md`` §1 and §7.
"""

from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]

BLUE = 0
RED = 1
TEAMS = (BLUE, RED)

AGENT_NAMES: tuple[str, ...] = ("blue_0", "blue_1", "red_0", "red_1")
NUM_AGENTS = len(AGENT_NAMES)

#: Team of each agent, indexed by the agent's position in ``AGENT_NAMES``.
AGENT_TEAMS: tuple[int, ...] = (BLUE, BLUE, RED, RED)


def other_team(team: int) -> int:
    """The opposing team index."""
    return RED if team == BLUE else BLUE


@dataclass(frozen=True, slots=True)
class CtfConfig:
    """Static description of a Capture the Flag board.

    The defaults describe the 7x5 board the repo's baselines are tuned on: small
    enough that a tabular learner is feasible, large enough that crossing the
    midline is a real commitment.
    """

    width: int = 7
    height: int = 5
    tag_radius: int = 1
    score_to_win: int = 3
    max_cycles: int = 256

    def __post_init__(self) -> None:
        if self.width < 4:
            raise ValueError("width must be at least 4 so both teams own territory")
        if self.height < 3:
            raise ValueError("height must be at least 3 so spawns and bases differ")
        if self.tag_radius < 0:
            raise ValueError("tag_radius must be non-negative")
        if self.score_to_win < 1:
            raise ValueError("score_to_win must be at least 1")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")

    # -- geometry ---------------------------------------------------------

    @property
    def bases(self) -> tuple[Position, Position]:
        """Base cell of each team, indexed by team."""
        row = self.height // 2
        return ((0, row), (self.width - 1, row))

    @property
    def spawns(self) -> tuple[Position, ...]:
        """Spawn cell of each agent, mirror-symmetric about the midline."""
        top = self.height // 4
        bottom = self.height - 1 - top
        return ((1, top), (1, bottom), (self.width - 2, top), (self.width - 2, bottom))

    def in_bounds(self, cell: Position) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def territory(self, cell: Position) -> int | None:
        """Owner of ``cell``: a team index, or ``None`` for the neutral midline."""
        x = cell[0]
        if x < self.width // 2:
            return BLUE
        if x > (self.width - 1) // 2:
            return RED
        return None

    def is_enemy_territory(self, cell: Position, team: int) -> bool:
        """True where ``team`` can be tagged: strictly inside the opponent's half."""
        owner = self.territory(cell)
        return owner is not None and owner != team

    def cells(self) -> list[Position]:
        """Every cell, in row-major order."""
        return [(x, y) for y in range(self.height) for x in range(self.width)]

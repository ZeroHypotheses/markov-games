"""Pygame rendering for Capture the Flag.

Both render modes share one draw path: :meth:`CtfRenderer.draw` paints a
``pygame.Surface``, and the mode decides what happens to it — blitted to a
window for ``human``, converted to an array for ``rgb_array``.

``rgb_array`` never touches the display subsystem, so it works headless (CI, and
the README GIF). Nothing here reads or writes game state; the renderer is a pure
function of :class:`CtfState`.
"""

from __future__ import annotations

import numpy as np
import pygame

from .config import AGENT_NAMES, AGENT_TEAMS, BLUE, NUM_AGENTS, RED, CtfConfig, Position
from .state import CtfState

# A restrained dark palette: territories are tinted, the agents are the only
# saturated things on the board, so the eye goes where the action is.
BACKGROUND = (17, 19, 27)
TERRITORY = ((31, 44, 72), (70, 34, 41))
NEUTRAL = (35, 38, 49)
GRID_LINE = (14, 16, 22)
TEAM_COLOR = ((94, 156, 255), (255, 99, 110))
TEAM_DARK = ((38, 66, 116), (112, 44, 50))
BASE_FILL = ((26, 38, 64), (62, 30, 36))
TEXT = (228, 233, 243)
MUTED = (124, 134, 156)
CARRIER_RING = (255, 214, 102)
MIDLINE = (72, 80, 100)

TEAM_LABELS = ("BLUE", "RED")


class CtfRenderer:
    """Draws a :class:`CtfState` onto a surface, a window, or an RGB array."""

    def __init__(
        self,
        config: CtfConfig,
        mode: str = "rgb_array",
        cell_size: int = 72,
        fps: int = 6,
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unknown render mode {mode!r}")
        self.config = config
        self.mode = mode
        self.cell = cell_size
        self.fps = fps

        self.margin = max(8, cell_size // 6)
        self.hud = max(56, cell_size)
        self.width = config.width * cell_size + 2 * self.margin
        self.height = config.height * cell_size + self.hud + 2 * self.margin

        pygame.font.init()
        self._font_large = pygame.font.Font(None, max(22, int(cell_size * 0.44)))
        self._font_small = pygame.font.Font(None, max(16, int(cell_size * 0.26)))
        self._font_agent = pygame.font.Font(None, max(16, int(cell_size * 0.34)))

        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        if mode == "human":
            pygame.init()
            pygame.display.set_caption("markov-games · capture the flag")
            self._window = pygame.display.set_mode((self.width, self.height))
            self._clock = pygame.time.Clock()

    # -- public API -------------------------------------------------------

    def render(self, state: CtfState) -> np.ndarray | None:
        """Render one frame. Returns an RGB array in ``rgb_array`` mode."""
        surface = self.draw(state)
        if self.mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

        assert self._window is not None and self._clock is not None
        pygame.event.pump()
        self._window.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.fps)
        return None

    def draw(self, state: CtfState) -> pygame.Surface:
        """Paint the whole frame: board, flags, agents, HUD."""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(BACKGROUND)
        self._draw_board(surface)
        self._draw_bases(surface)
        self._draw_flags(surface, state)
        self._draw_agents(surface, state)
        self._draw_hud(surface, state)
        return surface

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        pygame.font.quit()

    # -- pieces -----------------------------------------------------------

    def _rect(self, cell: Position) -> pygame.Rect:
        x, y = cell
        return pygame.Rect(
            self.margin + x * self.cell,
            self.margin + self.hud + y * self.cell,
            self.cell,
            self.cell,
        )

    def _center(self, cell: Position) -> tuple[int, int]:
        return self._rect(cell).center

    def _draw_board(self, surface: pygame.Surface) -> None:
        for cell in self.config.cells():
            owner = self.config.territory(cell)
            base = NEUTRAL if owner is None else TERRITORY[owner]
            # A faint checkerboard keeps distances readable without a grid that shouts.
            shade = 5 if (cell[0] + cell[1]) % 2 else 0
            pygame.draw.rect(surface, tuple(min(c + shade, 255) for c in base), self._rect(cell))
        for x in range(self.config.width + 1):
            left = self.margin + x * self.cell
            pygame.draw.line(
                surface,
                GRID_LINE,
                (left, self.margin + self.hud),
                (left, self.height - self.margin),
            )
        for y in range(self.config.height + 1):
            top = self.margin + self.hud + y * self.cell
            right = self.width - self.margin
            pygame.draw.line(surface, GRID_LINE, (self.margin, top), (right, top))
        self._draw_midline(surface)

    def _draw_midline(self, surface: pygame.Surface) -> None:
        """Where blue's half ends. On an even-width board this is the only cue."""
        x = self.margin + (self.config.width / 2) * self.cell
        dash = max(4, self.cell // 6)
        top = self.margin + self.hud
        bottom = self.height - self.margin
        for y in range(int(top), int(bottom), dash * 2):
            pygame.draw.line(surface, MIDLINE, (x, y), (x, min(y + dash, bottom)), 2)

    def _draw_bases(self, surface: pygame.Surface) -> None:
        for team in (BLUE, RED):
            rect = self._rect(self.config.bases[team]).inflate(-self.cell // 6, -self.cell // 6)
            pygame.draw.rect(surface, BASE_FILL[team], rect, border_radius=self.cell // 8)
            pygame.draw.rect(surface, TEAM_COLOR[team], rect, width=2, border_radius=self.cell // 8)

    def _draw_flags(self, surface: pygame.Surface, state: CtfState) -> None:
        for team in (BLUE, RED):
            carrier = state.carried_by[team]
            cell = state.flag_position(self.config, team)
            if carrier is None:
                self._draw_flag(surface, self._center(cell), team, scale=1.0)
            else:
                # Carried: tucked to the upper-right of the carrier.
                cx, cy = self._center(cell)
                self._draw_flag(
                    surface,
                    (cx + int(self.cell * 0.32), cy - int(self.cell * 0.20)),
                    team,
                    scale=0.92,
                )

    def _draw_flag(
        self, surface: pygame.Surface, center: tuple[int, int], team: int, scale: float
    ) -> None:
        cx, cy = center
        height = self.cell * 0.42 * scale
        top = cy - height / 2
        pole = max(2, int(self.cell * 0.035 * scale))
        pygame.draw.line(surface, TEXT, (cx, top), (cx, cy + height / 2), pole)
        flag = [
            (cx + pole / 2, top),
            (cx + self.cell * 0.26 * scale, top + height * 0.22),
            (cx + pole / 2, top + height * 0.44),
        ]
        pygame.draw.polygon(surface, TEAM_COLOR[team], flag)

    def _draw_agents(self, surface: pygame.Surface, state: CtfState) -> None:
        radius = int(self.cell * 0.30)
        for agent in range(NUM_AGENTS):
            team = AGENT_TEAMS[agent]
            center = self._center(state.positions[agent])
            if state.carrying(agent) is not None:
                pygame.draw.circle(surface, CARRIER_RING, center, radius + max(3, radius // 5))
            pygame.draw.circle(surface, TEAM_DARK[team], center, radius + 2)
            pygame.draw.circle(surface, TEAM_COLOR[team], center, radius)
            label = self._font_agent.render(AGENT_NAMES[agent][-1], True, BACKGROUND)
            surface.blit(label, label.get_rect(center=center))

    def _draw_hud(self, surface: pygame.Surface, state: CtfState) -> None:
        top = self.margin
        mid = top + self.hud // 2

        for team, align in ((BLUE, "left"), (RED, "right")):
            label = self._font_small.render(TEAM_LABELS[team], True, MUTED)
            score = self._font_large.render(str(state.scores[team]), True, TEAM_COLOR[team])
            if align == "left":
                surface.blit(label, label.get_rect(midleft=(self.margin, mid - self.hud // 5)))
                surface.blit(score, score.get_rect(midleft=(self.margin, mid + self.hud // 6)))
            else:
                right = self.width - self.margin
                surface.blit(label, label.get_rect(midright=(right, mid - self.hud // 5)))
                surface.blit(score, score.get_rect(midright=(right, mid + self.hud // 6)))

        clock = f"t {state.step:>4} / {self.config.max_cycles}"
        text = self._font_small.render(clock, True, MUTED)
        surface.blit(text, text.get_rect(center=(self.width // 2, mid - self.hud // 5)))

        # Time bar: how much of the episode is left, at a glance.
        bar = pygame.Rect(0, 0, int(self.width * 0.34), max(3, self.hud // 14))
        bar.center = (self.width // 2, mid + self.hud // 6)
        pygame.draw.rect(surface, NEUTRAL, bar, border_radius=bar.height // 2)
        progress = min(state.step / self.config.max_cycles, 1.0)
        filled = bar.copy()
        filled.width = max(1, int(bar.width * (1.0 - progress)))
        pygame.draw.rect(surface, MUTED, filled, border_radius=bar.height // 2)

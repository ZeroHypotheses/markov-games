"""Run episodes and — optionally — record them.

``mg-rollout`` is the smoke test you can watch: four random agents on the default
board, either in a window or written out as a GIF for the README.

    uv run mg-rollout --render human --episodes 2
    uv run mg-rollout --seed 748 --max-cycles 77 --cell-size 44 \
        --gif docs/media/capture-the-flag.gif
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .games.capture_the_flag import CaptureTheFlagEnv, CtfConfig
from .policies import RandomPolicy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """What one episode did. Frames are collected only when rendering to arrays."""

    steps: int
    scores: tuple[int, int]
    returns: dict[str, float]
    frames: tuple[np.ndarray, ...] = ()


def run_episode(
    env: CaptureTheFlagEnv,
    policy: RandomPolicy,
    seed: int | None = None,
    collect_frames: bool = False,
) -> EpisodeResult:
    """Play one episode to termination or truncation."""
    observations, _ = env.reset(seed=seed)
    returns = dict.fromkeys(env.possible_agents, 0.0)
    frames: list[np.ndarray] = []

    def capture() -> None:
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    capture()
    while env.agents:
        actions = policy(observations)
        observations, rewards, terminations, truncations, _ = env.step(actions)
        for agent, reward in rewards.items():
            returns[agent] += reward
        capture()
        if any(terminations.values()) or any(truncations.values()):
            break

    state = env.game_state
    return EpisodeResult(
        steps=state.step, scores=state.scores, returns=returns, frames=tuple(frames)
    )


def save_gif(path: str, frames: tuple[np.ndarray, ...], fps: int) -> None:
    """Write frames to an animated GIF. Requires the dev dependency ``imageio``."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit("GIF export needs imageio: `uv sync --group dev`, or drop --gif.") from exc
    imageio.mimsave(path, list(frames), fps=fps, loop=0)
    print(f"wrote {path} ({len(frames)} frames)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run random-agent Capture the Flag episodes.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", choices=["human", "rgb_array", "none"], default="none")
    parser.add_argument("--gif", metavar="PATH", help="record the first episode to a GIF")
    parser.add_argument("--width", type=int, default=CtfConfig().width)
    parser.add_argument("--height", type=int, default=CtfConfig().height)
    parser.add_argument("--max-cycles", type=int, default=CtfConfig().max_cycles)
    parser.add_argument("--cell-size", type=int, default=72, help="pixels per grid cell")
    args = parser.parse_args(argv)

    render_mode = "rgb_array" if args.gif and args.render == "none" else args.render
    config = CtfConfig(width=args.width, height=args.height, max_cycles=args.max_cycles)
    mode = None if render_mode == "none" else render_mode
    env = CaptureTheFlagEnv(config=config, render_mode=mode, cell_size=args.cell_size)

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            policy = RandomPolicy(env.possible_agents, seed=seed)
            result = run_episode(
                env,
                policy,
                seed=seed,
                collect_frames=bool(args.gif) and episode == 0 and render_mode == "rgb_array",
            )
            print(
                f"episode {episode}  seed {seed:>3}  steps {result.steps:>4}  "
                f"blue {result.scores[0]} - {result.scores[1]} red"
            )
            if args.gif and result.frames:
                save_gif(args.gif, result.frames, fps=int(env.metadata["render_fps"]))
    finally:
        env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

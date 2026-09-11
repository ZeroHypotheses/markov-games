# AGENTS.md — markov-games

Canonical instructions for **any** coding agent working in this repository
(Claude Code, Codex, Cursor, Copilot, Gemini CLI, OpenCode/Pi, …).
`CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` are symlinks to
this file. Edit this file; never edit the symlinks.

**This repo is harness-agnostic by construction.** Every instruction lives in
plain markdown that any agent can read — this file, `workflows/`, and `docs/`.
Nothing here requires a particular tool, and no harness gets behaviour another
cannot reproduce. Where a harness-specific file exists (`.claude/`, `CLAUDE.md`,
`GEMINI.md`), it is a **symlink or a thin pointer**, never a second copy of the
rules.

---

## 1. What this repo is

A hands-on **learning and experimentation repo** for multi-agent reinforcement
learning: self-play, coordination, competition, and game-theoretic learning,
studied through small Markov (stochastic) games we implement ourselves.

Two things are true at once, and both constrain the code:

1. **It is a learning repo.** The product is understanding. Algorithms are
   written from scratch in PyTorch so the mechanics are visible. A clever
   abstraction that hides the update rule is a regression, not progress.
2. **It is a public portfolio project.** It should be clean, tested,
   reproducible, well-rendered and pleasant to read for a stranger on GitHub.

The first — and currently only — game is **2v2 Capture the Flag**. Its rules are
specified in [`docs/capture-the-flag.md`](docs/capture-the-flag.md), which is the
single source of truth for the mechanics. Code and spec disagreeing is a bug in
one of them; fix both in the same change.

### Roadmap

Environment first, then baselines in this deliberate order:

```
Random  →  Tabular IQL  →  Independent DQN  →  IPPO  →  MAPPO
```

IQL is intentionally first. Its job is **not** to play Capture the Flag well —
it is to expose the classic MARL failure modes (non-stationarity, weak
coordination, unstable strategies, no role specialisation) on a problem small
enough to inspect by hand. Keep the tabular state representation feasible for as
long as that lesson is still being taught; move to Independent DQN when it
genuinely stops fitting.

The comparison that this repo is ultimately for: does centralised training
(MAPPO) buy coordination that independent learners never find, and do
attacker/defender roles emerge without being designed in?

---

## 2. Layout

```
markov-games/
├── AGENTS.md              ← you are here (canonical agent contract)
├── docs/
│   └── capture-the-flag.md  the rule spec — source of truth for the mechanics
├── src/markov_games/
│   ├── games/
│   │   └── capture_the_flag/
│   │       ├── config.py        CtfConfig — all tunable geometry, one place
│   │       ├── state.py         CtfState — the global state, frozen & hashable
│   │       ├── rules.py         the pure joint-action transition function
│   │       ├── observations.py  global state → per-agent observations
│   │       ├── env.py           PettingZoo ParallelEnv wrapper
│   │       └── render.py        Pygame renderer (human + rgb_array)
│   ├── policies.py        simple non-learning policies (Random)
│   └── rollout.py         run episodes, optionally record a GIF
├── tests/                 game mechanics, determinism, PettingZoo API, rendering
├── workflows/             harness-neutral playbooks (the real instructions)
├── scripts/               small utilities (harness-agnosticism check)
├── .agents/skills/        thin skill wrappers — canonical, cross-harness
└── .claude/skills/        symlinks into .agents/skills/ (Claude Code discovery)
```

---

## 3. Operating principles

These override default agent habits.

**P1 — The transition is a pure function of the joint action.** `rules.transition`
takes `(config, state, joint_action)` and returns the successor state. It never
mutates state in agent order; a rule that would let `blue_0` act "before"
`red_1` is wrong by construction. Simultaneity is the whole point of a Markov
game — if agent order can change the outcome, the bug is in the resolution
order, not in the caller.

**P2 — Mechanics are explicit, not emergent.** Every rule — collisions, swaps,
contested cells, tagging, respawns, flag drops, scoring — is stated in
`docs/capture-the-flag.md` and implemented as readable code with the same
vocabulary. No rule should have to be reverse-engineered from a loop.

**P3 — Implement the learning ourselves.** IQL, DQN, IPPO and MAPPO get written
here in NumPy/PyTorch, small and legible, each mapping visibly to the update
rule it implements. Do not pull in a MARL framework, and do not paste in an
existing PPO implementation. Copying working code teaches nothing; that is the
entire reason this repo exists.

**P4 — Determinism is a feature.** Same seed plus same actions ⇒ byte-identical
trajectory. All stochasticity lives in policies and in `reset(seed=…)`, never in
the dynamics. Every test asserts on exact states, not on tolerances.

**P5 — Global state and observations are different objects.** `CtfState` is what
the world knows; `observations.py` is what an agent is told. They are separate
today even though observability is (near-)full, so that partial observability
later is a change in one file — not a refactor.

**P6 — Smallest thing that answers the question.** Prefer a 40-line module over a
class hierarchy, a dataclass over a config framework, a function over a plugin
point. Add an abstraction when the *second* real case arrives, not when the
first one is imagined. No speculative interfaces, no speculative skills.

**P7 — Never report a number you did not produce.** Results come from a run, with
its seed and command recorded. Report over several seeds — single-seed MARL
results are noise. A negative result is a result; write it down.

---

## 4. Workflows

**The playbooks in `workflows/` are the instructions.** They are plain markdown
with no harness-specific syntax. Read the relevant one in full before acting —
whatever agent you are.

| Task | Playbook | Ask for it by saying |
|---|---|---|
| Add a new Markov game, or change the rules of one | [workflows/new-game.md](workflows/new-game.md) | "add a new game", "change the tagging rule" |
| Add a learning baseline (IQL, DQN, IPPO, MAPPO) | [workflows/algorithm.md](workflows/algorithm.md) | "implement IQL", "add the next baseline" |

### Skills

`.agents/skills/<name>/SKILL.md` holds a thin wrapper per workflow, in the
cross-harness skills layout. Each is a short file whose whole body says "read
`workflows/<x>.md`" — **the substance lives in one place only, so nothing can
drift.**

`.claude/skills/` contains symlinks into `.agents/skills/`, so Claude Code
offers them as `/mg-game` and `/mg-algorithm`. Other harnesses that read
`.agents/skills/` pick them up directly.

**If your harness has no skill mechanism at all, nothing is lost** — the table
above is the routing, and the playbook files are the same ones the skills point
to. A slash command is a shortcut, never a prerequisite.

**Editing rule.** To change *behaviour*, edit `workflows/*.md`. To change when a
skill triggers, edit `.agents/skills/*/SKILL.md` — the real files. Never replace
a `.claude/skills/` symlink with a copy, and never paste a playbook's content
into a skill; the moment the same instruction exists twice, the two harnesses
stop agreeing.

**Do not add a skill for a workflow that does not exist yet.** Two is the right
number today.

---

## 5. Conventions

**Python.** `uv` manages the environment. `uv sync` to install, `uv run …` to
execute. Python 3.11+. The package is a `src/` layout installed in editable mode.

```bash
uv sync                     # install
uv run pytest               # tests
uv run ruff check . && uv run ruff format --check .
uv run mg-rollout --render human           # watch four random agents
uv run mg-rollout --gif docs/media/ctf.gif # record for the README
```

**Types and style.** Type hints on every public function. `ruff` with
`line-length = 100` is the formatter and linter; do not hand-format around it.
Prefer `tuple[int, int]` positions over ad-hoc classes, frozen dataclasses over
mutable state, and module-level constants over magic numbers.

**Geometry.** Positions are `(x, y)` with `x` the column and `y` the row, origin
top-left, so `UP` decreases `y`. Agent order is fixed and global:
`("blue_0", "blue_1", "red_0", "red_1")`; team `0` is blue, team `1` is red.
Anything indexed per agent uses that order.

**Tests are the specification.** A mechanic without a test that pins its exact
outcome is not implemented. Construct the `CtfState` you want directly and call
`rules.transition` — do not drive the environment through dozens of steps to
reach an interesting position. Tests never sleep, never sample without a seed,
and never assert "roughly".

**Rendering.** `rgb_array` must work headless (no display, no `pygame.init()`),
because CI renders it and the README GIF comes from it. `human` is for people.

**Commits.** Present tense, scoped by area: `ctf: block swaps through the
midline`, `iql: add tabular baseline`, `docs: state the respawn tie-break`.

**Harness-agnosticism is checked.** `./scripts/check-harness-agnostic.sh`
verifies the invariants: contract files are symlinks not copies, skills are
canonical in `.agents/skills/` with `.claude/skills/` symlinked to them, skills
stay thin, every workflow is reachable from this file, and `workflows/` names no
particular tool. Run it after touching any agent-facing file.

---

## 6. Hard rules

- Never resolve a joint action by mutating state one agent at a time (P1).
- Never change a game rule in code without changing `docs/capture-the-flag.md`
  in the same commit, and vice versa.
- Never add reward shaping to Capture the Flag. The reward is ±1 on a capture
  and zero everywhere else; sparse credit assignment is the object of study, not
  an obstacle to route around.
- Never introduce randomness into the dynamics. If a rule needs a tie-break,
  make it a documented deterministic ordering.
- Never add a MARL framework, or copy an existing PPO/MAPPO implementation (P3).
- Never report an experimental result you did not run (P7).
- Never commit large binaries. The README GIF is the exception, and it stays
  small (a few hundred KB, not megabytes).

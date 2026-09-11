# Workflow: Add a learning baseline

The progression is fixed and deliberate:

```
Random  →  Tabular IQL  →  Independent DQN  →  IPPO  →  MAPPO
```

Take the next one. Skipping ahead — "let's just do MAPPO" — costs the entire
lesson, which is *why* each step was needed: what independent learners fail at,
and what centralised training actually buys.

Each baseline is written here, from the update rule, small enough to read in one
sitting. Do not install a MARL library and do not paste in someone's PPO.

## Steps

**1. State what this baseline is supposed to teach**, before any code, in the
algorithm module's docstring: the update rule it implements, what it is expected
to do on this game, and what it is expected to fail at. For IQL the expected
failure is the point — non-stationarity from a moving co-learner, unstable
strategies, no attacker/defender specialisation.

**2. Write the update rule, visibly.** One file per algorithm, under
`src/markov_games/algorithms/`. The learning step should be readable as the
equation it came from, with the equation in a comment above it. Prefer an
explicit loop over a clever vectorisation when the loop is what makes the
mechanism legible — this is a learning repo, and the algorithms are the exhibit.

**3. Share nothing between agents unless the algorithm says to.** Independent
learners get independent parameters, independent replay, independent optimisers.
Centralised critics get the global state from `observations.py`, and that is the
*only* extra information they get. If you find yourself passing a teammate's
observation into an "independent" learner, stop — that is the experiment you
were meant to run, not a convenience.

**4. Make the run reproducible.** Seed everything (Python, NumPy, PyTorch);
record the seed, the config and the exact command with the results. Same seed ⇒
same curve.

**5. Evaluate against the ladder below it.** A new baseline reports win rate,
capture rate and episode length against: random, and every baseline already
implemented. Several seeds, always — a single-seed MARL result is noise.

**6. Write down what happened, including when it failed.** A short note with the
numbers you actually produced, the command that produced them, and what you now
believe about the algorithm. Falsified expectations are the most valuable entries.

## Questions worth keeping in view

These are what the ladder exists to answer. Every baseline should say something
about at least one:

- Does the tabular representation stay feasible, and where exactly does it break?
- Do independent learners destabilise each other, and can you see it in the
  curves rather than infer it?
- Do attacker and defender roles emerge without being designed in?
- Does centralised training improve coordination, or only sample efficiency?
- Does self-play converge to anything, or cycle?

## Rules

- Implement the learning logic yourself; no framework, no copied implementation.
- No reward shaping to make a baseline look better. Change the algorithm, or
  report that it failed.
- Never report a number that did not come out of a run you performed.
- Keep training scripts small and flat. Hyperparameters are a dataclass with
  defaults, not a config framework.

# Workflow: Add a Markov game (or change the rules of one)

A game in this repo is five small modules and a rule spec. Capture the Flag is
the reference implementation — read
[`src/markov_games/games/capture_the_flag/`](../src/markov_games/games/capture_the_flag/)
before writing a second game, and copy its shape.

**The rule spec comes before the code.** Not as ceremony: writing the rules in
prose is what surfaces the ambiguous cases (two agents into one cell, a swap, a
tag on the step a flag is grabbed) while they are still cheap to decide.

## Steps

**1. Write the spec first.** `docs/<game>.md`, stating for every mechanic what
happens, in terms a reader can check the code against. It must answer, at
minimum:

- What is the state? What is the joint action?
- How is a *simultaneous* joint action resolved? Contested cells, swaps,
  cycles, walls.
- What are the deterministic tie-breaks, and why is each one fair to both teams?
- When does an episode terminate? When is it truncated?
- What exactly is rewarded?

If a question has no answer yet, write the question down in the spec rather than
letting the code decide by accident.

**2. Lay out the modules.** One concern each, in dependency order:

```
<game>/
├── config.py        a frozen dataclass: geometry, limits, scores. No logic.
├── state.py         a frozen, hashable dataclass + tiny derived helpers.
├── rules.py         transition(config, state, joint_action) -> (state, events)
├── observations.py  state -> per-agent observation, and the global state vector
├── env.py           the PettingZoo ParallelEnv wrapper. Bookkeeping only.
└── render.py        Pygame: human + rgb_array from the same draw path.
```

`rules.py` is pure: no RNG, no I/O, no mutation of its arguments, no reference to
the environment object. That is what makes the mechanics testable in isolation
and the dynamics reproducible.

**3. Resolve the joint action as a fixpoint, never in agent order.** Compute
every agent's intent, then repeatedly cancel the intents that conflict until
nothing changes. Cancelling is monotone, so it terminates. Any rule whose
outcome depends on which agent is examined first is a bug — write the test that
proves it is order-independent (permute the agents, expect the same successor).

**4. Keep the global state and the observation apart** even when observability is
full. `observations.py` is the only place that decides what an agent is told.

**5. Test each rule as a unit.** Build the exact `state` that isolates the
mechanic, call `transition` once, assert the exact successor. One test per rule
in the spec, named after it. Then the whole-game tests: determinism under a
seed, and the PettingZoo parallel API conformance test.

**6. Render it from day one.** A game you cannot watch is a game whose bugs you
will not see. `rgb_array` must work with no display available; `human` shares the
same draw code.

**7. Wire it up.** Export the environment from `src/markov_games/__init__.py`,
add it to the README's table of games, and make sure a random rollout runs and
records a GIF.

## Changing the rules of an existing game

Same discipline, smaller loop:

1. Change `docs/<game>.md` first, in the same commit as the code.
2. Change `rules.py`.
3. Update the test that pinned the old behaviour — never delete it; a rule that
   changed should have a test that changed with it.
4. Say out loud, in the commit message, what this does to results already
   recorded: a rule change invalidates every baseline measured before it.

## Rules

- The transition function stays pure and order-independent.
- No randomness in the dynamics. Ties are broken by a documented, fixed order.
- No reward shaping. Sparse rewards are the object of study.
- Every constant that a reader might want to change belongs in `config.py`,
  typed and defaulted, not inlined at its use site.

---
name: mg-game
description: Add a new Markov game to this repo, or change the rules of an existing one (Capture the Flag). Trigger when the user says: add a game, new environment, change the tagging rule, the collision rule is wrong, implement <game> as an env. Do NOT trigger for learning algorithms (use mg-algorithm).
---

# mg-game

**This skill is a wrapper. The instructions live in a harness-neutral playbook
so every agent in this repo behaves the same way.**

Read and follow [`workflows/new-game.md`](../../../workflows/new-game.md) in full,
plus the operating principles in [`AGENTS.md`](../../../AGENTS.md).

The rule spec in `docs/` is written or updated *before* the code, in the same
change — the prose is what forces the ambiguous simultaneity cases into the open.

The transition function stays pure and order-independent: compute the successor
from the full joint action, never by mutating state agent by agent.

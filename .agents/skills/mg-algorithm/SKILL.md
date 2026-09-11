---
name: mg-algorithm
description: Implement the next MARL learning baseline in the ladder Random to IQL to Independent DQN to IPPO to MAPPO, and evaluate it against the ones below it. Trigger when the user says: implement IQL, add the next baseline, train agents on CTF, compare IPPO and MAPPO. Do NOT trigger for environment or rule changes (use mg-game).
---

# mg-algorithm

**This skill is a wrapper. The instructions live in a harness-neutral playbook
so every agent in this repo behaves the same way.**

Read and follow [`workflows/algorithm.md`](../../../workflows/algorithm.md) in
full, plus the operating principles in [`AGENTS.md`](../../../AGENTS.md).

Take the next rung of the ladder, not the most interesting one. Write the update
rule from the equation — no MARL framework, no copied PPO.

State what the baseline is expected to fail at before running it, and report the
failure honestly over several seeds when it happens.

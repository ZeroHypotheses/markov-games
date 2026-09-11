# 2v2 Capture the Flag — rule specification

This document is the **source of truth** for the mechanics. The code in
[`src/markov_games/games/capture_the_flag/`](../src/markov_games/games/capture_the_flag/)
implements exactly what is written here. If the two disagree, one of them is a
bug — fix both in the same change.

Everything below is deterministic. Given a state and a joint action there is
exactly one successor state; no dice are rolled anywhere in the dynamics.

---

## 1. The board

A `width × height` grid, default **7 × 5**, both configurable. Positions are
`(x, y)` with `x` the column and `y` the row, origin **top-left**, so `UP`
decreases `y`.

Territory is split vertically and symmetrically:

| Region | Columns (general) | Columns (7 × 5) |
|---|---|---|
| Blue territory | `x < width // 2` | 0, 1, 2 |
| Red territory | `x > (width - 1) // 2` | 4, 5, 6 |
| Neutral | everything else | 3 |

With an odd width there is a one-column neutral midline; with an even width the
two territories meet and there is none. Both cases are symmetric.

**Bases.** Blue's base is `(0, height // 2)`, red's is `(width - 1, height // 2)`.
A team's flag sits on its own base when it is not being carried.

**Spawns.** Four distinct cells, mirror-symmetric about the vertical midline:

```
blue_0 (1, height // 4)              red_0 (width - 2, height // 4)
blue_1 (1, height - 1 - height // 4) red_1 (width - 2, height - 1 - height // 4)
```

On a 7 × 5 board: `blue_0 (1,1)`, `blue_1 (1,3)`, `red_0 (5,1)`, `red_1 (5,3)`.

**Agents.** Exactly four, in the fixed global order
`("blue_0", "blue_1", "red_0", "red_1")`. Team 0 is blue, team 1 is red.

---

## 2. State

The full state is `(positions, carried_by, scores, step)`:

- `positions` — one `(x, y)` per agent, in agent order. **At most one agent per
  cell**, always.
- `carried_by[team]` — the index of the agent carrying *that team's* flag, or
  `None` if the flag is home. Only an opponent can carry a flag, so
  `carried_by[0]` is always a red agent or `None`.
- `scores[team]` — captures made by that team.
- `step` — elapsed timesteps.

Flag positions are **derived**, never stored: a flag is at its owner's base if
`carried_by` is `None`, otherwise at its carrier's position. There is no
"dropped on the ground" state — see §5.

## 3. Actions

Each agent picks one of five actions, simultaneously and independently:

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `STAY` | `UP` (`y-1`) | `DOWN` (`y+1`) | `LEFT` (`x-1`) | `RIGHT` (`x+1`) |

The successor state is computed **from the full joint action at once**. No agent
moves "before" another; there is no agent-order dependence anywhere in §4–§7.

---

## 4. Movement resolution

**Step 1 — intent.** Each agent's target cell is its current cell plus its action
offset. A target outside the grid is clipped to the current cell: *walking into a
wall is a no-op*, not an error.

**Step 2 — cancellation to a fixpoint.** Repeatedly apply both rules below until
a full pass changes nothing. A cancelled ("blocked") agent's target becomes its
current cell, which can cascade into further cancellations.

- **R1 — Swap.** Two agents that would exchange cells both stay. You cannot move
  through an opponent, and a swap is never resolved in anyone's favour.
- **R2 — Contested cell.** If two or more agents have the same target cell, *all*
  of them stay. Nobody wins a contested cell — including the case where one of
  the claimants is an agent that is standing still, which is how "you cannot walk
  into an occupied cell" falls out.

Cancellation is monotone (a blocked agent never becomes unblocked), so the
fixpoint is reached in at most four passes, and the result does not depend on
the order in which agents are examined.

**Consequences worth stating explicitly:**

- A **chain** resolves correctly: if `A → B → C` and `C` stays, `B` is blocked by
  R2, and then `A` is blocked by R2 against the now-stationary `B`.
- A **rotation** succeeds: if `A → B → C → A` in a cycle, every target is
  distinct and no pair swaps, so all of them move. A cycle of agents rotating
  through occupied cells is legal.
- Two agents can never occupy the same cell, so every later rule in this document
  operates on distinct positions.

---

## 5. Tagging, flags and scoring

These are all computed from the **post-movement** positions, simultaneously.

**Step 3 — tagging.** An agent is tagged if and only if:

1. it is standing in **enemy territory** (neutral cells and its own territory are
   safe), and
2. at least one opponent is within **Manhattan distance ≤ `tag_radius`**
   (default 1, configurable).

An agent standing in its own half is never tagged, so defenders are always safe
at home and attackers always take a risk. On an even-width board where the two
territories touch, two agents that have each crossed the midline can tag each
other on the same step; this is symmetric and intended.

**Step 4 — flag drop.** If a tagged agent was carrying a flag, that flag
**returns instantly to its owner's base**. There is no loose flag lying on the
ground: this keeps the state space small enough for a tabular baseline to be
legible. (A ground-drop variant is a deliberate future extension, not an
oversight.)

**Step 5 — respawn.** Every tagged agent returns to its own spawn cell. If that
cell is occupied — by an untagged agent, or by an agent that respawned first —
it is placed on the nearest free cell, ordered by Manhattan distance from its
spawn and then by `(y, x)`. Simultaneously tagged agents are placed in the fixed
global agent order. *This ordering is a tie-break for placement only; it is
documented rather than arbitrary, and no agent gains or loses anything by it.*

**Step 6 — pickup.** An untagged agent standing on the **enemy base** picks up the
enemy flag, if that flag is home and not already carried. Since at most one agent
occupies a cell, pickup is never contested. A tagged agent cannot pick up on the
step it was tagged.

**Step 7 — capture.** An untagged agent carrying the enemy flag and standing on
**its own base** scores: its team's score increases by one, and the captured flag
returns to its owner's base. Agents keep their positions; the world is not reset.

Carrying its own team's flag is impossible, and an agent can carry at most one
flag, so these two steps never interact within a single timestep.

---

## 6. Rewards

Sparse, shared within a team, and zero-sum:

| Event | Both agents of the scoring team | Both agents of the other team |
|---|---|---|
| A capture | **+1** | **−1** |
| Anything else | 0 | 0 |

There is no shaping: no distance-to-flag bonus, no tag reward, no step penalty.
Sparse credit assignment is what this environment is for.

If both teams somehow capture on the same timestep, the rewards cancel to zero
for everyone, which is correct.

---

## 7. Termination and truncation

- **Terminated** — a team's score reaches `score_to_win` (default 3). All four
  agents terminate together.
- **Truncated** — `step` reaches `max_cycles` (default 256) without a winner. All
  four agents truncate together.

A timestep that both scores the winning capture and hits `max_cycles` counts as
terminated, not truncated.

---

## 8. Observations

Observations are kept **conceptually separate from the state** even though they
are currently near-complete, so that partial observability is a change in one
file rather than a refactor.

Each agent receives a `float32` vector, ordered **relative to itself** so that the
two teams see a structurally identical problem:

| Slice | Contents |
|---|---|
| 0–2 | self: `x/(w-1)`, `y/(h-1)`, carrying flag (0/1) |
| 3–5 | teammate: same three |
| 6–8 | first opponent: same three |
| 9–11 | second opponent: same three |
| 12–14 | own flag: `x`, `y` normalised, stolen (0/1) |
| 15–17 | enemy flag: `x`, `y` normalised, carried by own team (0/1) |
| 18 | score difference, `(own − enemy) / score_to_win` |
| 19 | time remaining, `1 − step / max_cycles` |
| 20 | side of the board: 1 for blue, 0 for red |

Opponents are listed in the fixed global order within their team, not sorted by
distance. Index 20 is what stops the relative encoding from being ambiguous about
which way the board runs.

The **global state vector** (`env.state()`, for centralised critics later) uses
absolute ordering instead: all four agents in global order, then blue's flag,
then red's, then both scores and the time remaining.

---

## 9. What is deliberately not here

Named so that nobody "fixes" them by accident:

- **No loose flags.** A dropped flag goes straight home (§5).
- **No respawn delay.** A tagged agent is back on its spawn on the same step.
- **No requirement that your own flag be home to capture.** Scoring needs only the
  enemy flag and your own base.
- **No walls or obstacles**, no partial observability, no communication channel,
  no stochastic transitions.

Each is a plausible next increment. Each would change results already measured,
so each needs a spec change, a test change, and a note in the commit.

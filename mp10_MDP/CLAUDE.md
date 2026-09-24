# CLAUDE.md — mp10_MDP

Value iteration on a grid-world MDP. Three functions, graded numerically
against stored solutions. See the repo-root `CLAUDE.md` for conventions.

## The shapes

| function | returns |
|---|---|
| `compute_transition_matrix(model)` | `P[r, c, a, r', c']`, an **M x N x 4 x M x N** array |
| `update_utility(model, P, U_current)` | one Bellman sweep, M x N |
| `value_iteration(model)` | converged utility, M x N |

Actions are `0=left, 1=up, 2=right, 3=down`.

## The model's own conventions

`model.D[r, c, k]` is the probability of the intended move going
`k=0` straight, `k=1` counter-clockwise, `k=2` clockwise — it is **relative to
the chosen action**, not absolute compass directions. `submitted.py` encodes
this as a per-action `movements` list; getting the rotation backwards produces
a transition matrix that is wrong only in the stochastic off-diagonal terms,
which is easy to miss.

Two boundary rules:

* Moving into a wall (`model.W`) or off the grid leaves the agent in place —
  add that probability back to `P[r, c, a, r, c]`.
* **Terminal states have all-zero transitions.** `model.T[r, c]` marks them;
  `submitted.py` sets `P[r, c, :, :, :] = 0` there rather than a self-loop.
  A self-loop makes value iteration diverge on the terminal reward.

## Grading

`tests/test_visible.py` loads `models/model_small.json` / `model_large.json`,
compares against `solution_small.json` / `solution_large.json`, and requires
`abs(diff).max() < 1e-2`. Four tests, 15 points each (P and U on each model).
The failure message prints the single worst-disagreeing index, which is the
fastest way to spot a rotation or boundary error.

`epsilon = 1e-3` at the top of `submitted.py` is the convergence threshold for
`value_iteration`.

## Setup note

**No `requirements.txt` in this directory.** It needs numpy, plus `matplotlib`
— `utils.py` imports `matplotlib.pyplot` at module scope, so the tests fail at
import time without it even though nothing plots during grading.

# CLAUDE.md — mp11_RL_with_PongGame

Tabular Q-learning for one-player Pong. This is the **required** half of MP11;
`../mp11_DeepQ_RL/` is the deep-Q extra-credit variant of the same assignment
and has its own, more detailed CLAUDE.md worth reading alongside this.

## What is graded

Five 8-point unit tests on the `q_learner` methods, plus a 10-point test that
loads `trained_model.npz` and plays: **average score over 10 games must exceed
6**. That last one is stochastic — a borderline model can fail one run and pass
the next.

## The state and the table

The learner is handed the **quantized** state: a list of 5 ints
`[ball_x, ball_y, ball_vx, ball_vy, paddle_y]` with cardinalities
`[10, 10, 2, 2, 10]`. `Q` and `N` are therefore `(10,10,2,2,10,3)` arrays; index
them with `tuple(state)` and an action index.

**Actions are -1 / 0 / +1 but table indices are 0 / 1 / 2.** Every method has to
convert, and mixing the two up is the most common bug here — it produces a
learner that trains without error and plays badly.
`report_exploration_counts` and `report_q` return length-3 arrays in index
order; `choose_unexplored_action`, `exploit` and `act` return the *action*.

## Contracts that are easy to miss

* `choose_unexplored_action` must **increment `N`** for the action it picks, and
  must choose uniformly among all under-explored actions — not the first one.
  It returns `None` once every action has been tried `nfirst` times.
* `q_local(reward, newstate)` is `reward + gamma * max_a Q[newstate, a]`; `learn`
  moves `Q` toward it by `alpha`.
* `self.flag` forces pure exploitation when set — used to evaluate without
  exploration noise.
* `pong.PongGame.run` returns **three** values for a `q_learner`, one otherwise.
  That branch is an identity check on `type(self.learner)`, so reloading
  `submitted` after building the learner silently changes the arity. Construct
  the learner after the last `importlib.reload`.

## Running

```bash
python grade.py
python pong.py --player q_learning          # train with a visible board
python pong.py --player human               # play it yourself
```
Needs pygame for the visible modes; `visible=False` avoids importing it.

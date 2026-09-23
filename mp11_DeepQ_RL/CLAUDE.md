# CLAUDE.md — mp11_DeepQ_RL

Guidance for Claude Code when working in this directory. The repo-root
`CLAUDE.md` covers conventions shared by every MP; this file covers what is
specific to the deep-Q extra credit.

## The one rule that bites

`submitted.py` must stay **self-contained**. Gradescope receives only
`submitted.py` and a checkpoint, so the network definitions, the replay buffer
and the config dataclass all live in that one file. Never refactor them into a
sibling module — `train_deepq.py`, `evaluate.py`, `visualize.py` and
`viz_style.py` may import *from* `submitted`, never the other way round.

Run everything from inside this directory; `pong.py` and the tests import
`submitted` as a top-level module and resolve paths against the CWD.

```bash
PY=../.venv/bin/python          # the venv the repo was set up with
$PY grade.py                    # the 5 extra-credit tests
$PY train_deepq.py --arch resnet --frames 600000
$PY evaluate.py --model trained_model.pkl --games 10 --max-frames 300000
$PY visualize.py all --model trained_model.pkl
```

## Contracts that are not obvious from the code

* **`pong.PongGame.run` returns 1 or 3 values** depending on
  `type(self.learner) == submitted.deep_q`. Reloading `submitted` after
  constructing the learner breaks that identity check and silently changes the
  return arity. Construct the learner *after* the last `importlib.reload`.
  `deepq.md` has the full write-up.
* **A negative reward is a terminal**, but the environment does not say so:
  `PongGame.update` silently respawns the ball in the same call. Bootstrapping
  across that reset makes the task look unlearnable.
* **`tests/test_extra.py` runs its ten games with no frame cap.** A learner
  that rarely misses makes the autograder run for hours. When choosing what to
  ship, weigh score against grading time — see "Which checkpoint to ship".
* **`learn()` is called on every frame during grading**, including after
  `load()`. That is why `load()` switches to evaluation mode (greedy, no
  gradient updates) instead of leaving exploration on.

## Evaluation numbers are only meaningful with their cap

Three different budgets are in play, and they are not comparable:

| where | games | frames per game |
|---|---|---|
| `train_deepq.py` periodic `eval` | 10 | `--eval-max-frames` (6,000) |
| `train_deepq.py` `final greedy eval` | 20 | `--final-max-frames` (30,000) |
| `evaluate.py` | `--games` | `--max-frames` (300,000+) |
| `tests/test_extra.py` | 10 | unbounded |

A game that ends because it ran out of frames is scored at its current value,
so **any average with capped games in it is a lower bound**. `evaluate.py`
always prints how many games hit the budget; quote that number alongside the
mean or the mean is misleading.

A good agent costs **~165 frames per hit**, so a score of 1,000 needs a budget
of ~165,000 frames and the transformer needed 6,000,000 to stop being truncated. This has bitten twice
already: the 30,000-frame cap made the transformer look like a score of 170
when it actually scores 8,755, and it ranked the mlp *above* the resnet when
the true order is the other way round. Caps do not just compress the top of
the scale, they reorder the table. Once a run's `eval` line shows `max` sitting
at `eval_max_frames / 165`, that evaluation has stopped measuring the policy —
raise the budget or compare with `evaluate.py` instead.

Evaluation is inference-only and roughly 10x faster per frame than training,
so large budgets are cheaper than they look.

## Episode boundaries

Three things must happen when an episode ends *or is truncated*: the frame
stack resets, the n-step queue is flushed, and no transition is allowed to
straddle the boundary. Call `agent.end_episode()` — do not poke `_frames` or
`_pending` directly. Truncation (hitting a frame budget) is not a terminal:
the queue is flushed but nothing is marked `done`, because the episode did not
actually end.

## What makes the agent good (measured, not guessed)

Probes on the mlp, all else equal. The two columns disagree, and that is the
whole point — **do not tune on an early snapshot.**

| configuration | at 250k frames | at 1.5M frames (`evaluate.py`) |
|---|---:|---:|
| gamma 0.99, 1-step — the original defaults | 7.0 | 36.2 |
| gamma 0.995, 3-step | 23.0 | **243.9** |
| gamma 0.999, 5-step | 7.7 | 171.8 |
| gamma 0.995, 3-step, `--shaping 1.0` | **83.7** | 160.8 |

* **gamma 0.99 is too small for this environment.** Its horizon is ~100 frames
  and a rally is ~165, so the agent could barely see its own next hit. Moving
  to 0.995 with 3-step returns is the one change that is clearly worth it at
  every horizon (~7x at 1.5M).
* **`--shaping` buys early speed, not a better final policy.** It is 12x ahead
  at 250k frames and then gives the lead back: by 1.5M its median (149.5) is
  indistinguishable from plain gamma+n-step (150.0), and the mean gap is
  tail-driven on 8 games. Use it to get a usable agent quickly; do not assume
  it raises the ceiling. It is potential-based (`tracking_potential`), a
  function of state alone and zero at terminals, so by Ng, Harada & Russell
  (1999) the optimal policy is provably unchanged — if you edit it, keep that
  property or you are optimising something else.
* **Never trust `best.pkl` or `last.pkl` alone -- which one wins flips.** On
  the 1.5M probes `best` beat `last` by 10-20x (late-training collapse); on the
  2.5M shaped resnet `last` beat `best` by 1.9x (2555 vs 1367), because "best"
  had been chosen from a 50,000-frame evaluation that saturates around 300.
  A cap does not merely mis-measure, it mis-*selects*. Score the `ckpt_*.pkl`
  snapshots with `evaluate.py` and pick from those.
* **Tuning does not transfer across backbone sizes.** The gamma/n-step recipe
  at 2.5M frames lifted the resnet 210 -> 2555 and sank the transformer
  8755 -> 437, with or without shaping. Re-measure per architecture; a win on
  the mlp is not evidence about the transformer.
* **Long runs oscillate wildly.** Scoring the no-shaping transformer's
  checkpoints 250k frames apart gave 336, 353, 375, **3**, **437**, 33, 30.
  Never quote a single end-of-run number for a 2.5M-frame run; score the
  snapshots and say which one you picked.
* Capacity dominates at the top end: transformer > resnet > mlp > cnn. No mlp
  probe passed ~250, while the transformer scores 8,755.

## Which checkpoint to ship

`--promote` picks the highest final mean, which is usually *not* what you want
in `trained_model.pkl`. Prefer a model that clears the graded threshold with a
comfortable margin and grades quickly; keep the strongest model beside it under
its own name. The README records the current choice and why.

## Costs, so you can plan a run

On CPU, single process, 8 threads, roughly: mlp ~1,500 frames/s, cnn ~800,
resnet ~345, transformer ~283. 600k frames is 7 min for the mlp and 35 for the
transformer. The runs are independent — launch them in parallel with a modest
`--threads` each rather than one after another. Long evaluations parallelise
too: `evaluate.py` plays each game in its own process.

## Things that are already fixed — do not "fix" them again

* `pong.py`'s `--player q_trained` branch needs `state_quantization = None`
  (it is a deep-Q model, trained on raw floats). The line is there with a
  comment.
* `TransformerBackbone` passes `enable_nested_tensor=False` on purpose;
  `norm_first=True` rules out that fast path and torch warns otherwise.
* `q_values_for_states` reconstructs a frame stack by running the ball
  backwards along its velocity. Repeating a single state instead describes a
  ball that is stationary and moving at once, and the plots go to mush.
* `alpha` is stored but unused for optimisation; Adam uses `DeepQConfig.lr`.

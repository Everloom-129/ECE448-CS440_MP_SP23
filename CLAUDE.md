# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Completed machine problems (MPs) for UIUC ECE448/CS440 (Artificial Intelligence), Spring 2023. Each `mpNN_*/` directory is an **independent, self-contained assignment** distributed by the course staff — there is no shared package, no top-level build, and no cross-MP imports. Dependencies, data, and graders are per-directory.

Per `README.md`, this code is reference material only; it must not be submitted for academic credit elsewhere.

## Working in an MP

Every command must be run **from inside the MP directory**, not the repo root. Graders and tests import `submitted` as a top-level module and resolve data paths relative to the CWD (`data/part-1/tiny`, `models/model_small.json`, `cifar10_batches/`, …), so running from the root will fail with import or file-not-found errors.

```bash
cd mp10_MDP
pip install -r requirements.txt   # mp03_KNN and mp10_MDP ship none (mp10 needs numpy + matplotlib via utils.py)
                                  # mp04 also ships environment.yml (conda, py3.9)
python grade.py                   # run the visible tests
python grade.py -j                # same, as Gradescope JSON (mp05/mp06 use --gradescope instead)
```

Run a single test with unittest directly (still from the MP directory):

```bash
python -m unittest tests.test_visible.TestStep.test_small_P -v
```

Test classes are `TestStep` in most MPs, `TestMP4` in mp04/mp09, `Test`/`grading_tests` in mp11/mp06.

Interactive/visual entry points (pygame):

```bash
cd mp05_Search && python main.py data/part-1/tiny --search bfs   # also astar_single, astar_multiple, fast; --human, --save out.png
cd mp06_Minimax_Game && python main.py --player0 human --player1 alphabeta --depth1 3
```

Each MP also has an `mpNN_notebook.ipynb` — the staff-written assignment writeup. It documents the expected signature and semantics of every function in `submitted.py`, runs them cell by cell against the real data, and ends with `!python grade.py`. Read it before changing a `submitted.py`; it is the authoritative spec for that MP.

## Structure of every MP

| File | Role |
|---|---|
| `submitted.py` | **The only file intended to be edited.** Holds the student implementation. |
| `grade.py` | Entry point; discovers `tests/` via `unittest.defaultTestLoader.discover('tests')` and runs it in text or Gradescope-JSON mode. |
| `tests/test_visible.py` | Staff tests. Weighted with `@weight(n)` / `@partial_credit` from `gradescope_utils`, which is why that package is in every `requirements.txt`. |
| `reader.py` / `utils.py` / `maze.py` / `models.py` / `pong.py` | Staff-provided support code — data loading, environment, model definitions. Treat as read-only. |
| `solution*.json`, `grading_examples/`, `key_s`, `trained_model.npz` | Precomputed reference answers the tests compare against. |

Tests import `submitted` directly and assert numeric closeness to a stored ground truth (e.g. `np.abs(P - P_gt).max() < 1e-2`), so function **names and return shapes in `submitted.py` are a fixed contract** — the tests break if they change. mp05 is the exception: its `grade.py` is hand-rolled (no `tests/` dir) and scores path validity, path length, and `maze.states_explored` against a pickled answer key (`key_s`, the student-mode key; the instructor key `key_i` is absent).

Some MPs carry extra graded surfaces: `mp08_Hidden_Markov_Model/tests/test_visible_ec.py` (extra credit, auto-discovered alongside the visible tests) and `mp06_Minimax_Game/mp06_extracredit.zip`.

## Per-MP guidance

Every `mpNN_*/` directory has its own `CLAUDE.md` with the contracts, shapes
and grading thresholds specific to it. **Read that first when working inside
one** — several MPs grade in ways this file does not describe (mp05 scores how
many states you explored, mp08 grades accuracy thresholds rather than a stored
answer, mp06 grades the search tree including what you pruned, mp04 enforces a
parameter-count band).

## MP index

`mp01` joint/marginal/conditional distributions · `mp02` naive Bayes text classification · `mp03` k-NN · `mp04` neural nets in PyTorch · `mp05` BFS / A* maze search · `mp06` minimax, alpha-beta, stochastic search over chess · `mp07` unification and forward chaining · `mp08` HMM POS tagging (Viterbi) · `mp09` CNN perception on CIFAR-10 · `mp10` MDP value iteration · `mp11` reinforcement learning on Pong.

`mp11_DeepQ_RL/` and `mp11_RL_with_PongGame/` are two variants of the same assignment: the latter is the tabular Q-learning version graded by `tests/test_visible.py`; the former is the deep-Q extension, graded by `tests/test_extra.py`, which loads a PyTorch `trained_model.pkl` and needs a 10-game average score above 20 for full credit. It has grown its own tooling (`train_deepq.py`, `evaluate.py`, `visualize.py`) and its own `mp11_DeepQ_RL/CLAUDE.md` — read that before working in there, as several of its contracts (evaluation caps, checkpoint choice, episode boundaries) are easy to get wrong.

## Conventions in `submitted.py`

- Unimplemented stubs in the staff skeleton `raise RuntimeError('You need to write this part!')`; solved ones keep the original docstring (parameters and output shapes) above the implementation.
- Borrowed ideas are cited in comments with a URL near the top of the file or the function. Keep that habit when adding code.
- NumPy is the default vocabulary; prefer vectorized operations, but correctness against the stored ground truth matters more than speed except in mp05, where `states_explored` is itself scored.

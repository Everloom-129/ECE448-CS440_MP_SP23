# CLAUDE.md — mp06_Minimax_Game

Minimax, alpha-beta and stochastic search over a real chess engine. See the
repo-root `CLAUDE.md` for shared conventions.

## The search functions return a triple

Every one of `minimax`, `alphabeta`, `stochastic` returns
**`(value, moveList, moveTree)`**:

* `value` — the heuristic value of the chosen line
* `moveList` — the principal variation, a list of `[from, to, promote]` moves
* `moveTree` — a **nested dict** recording every move the search actually
  examined, keyed by `encode(*move)`

`moveTree` is graded, not just the value. `tests/test_visible.py` walks it
recursively against `grading_examples/*.json` and reports the first path where
they differ. This means **move ordering and pruning behaviour are part of the
answer** — an alpha-beta that prunes a different (even if equally valid) set of
branches fails. Generate moves in the order `generateMoves` yields them.

## The engine is a package, not a helper file

`from chess.lib.utils import encode, decode, initBoardVars`,
`from chess.lib.heuristics import evaluate`, `from chess.lib.core import makeMove`.
State is the triple `(side, board, flags)`; `side` is the player to move.
`evaluate(board)` is the heuristic — do not write your own.

Test positions come from `res/savedGames/game0.txt` and `game1.txt`, replayed
through `makeMove` by the `load_game` helper in the test file.

## Determinism in `stochastic`

`stochastic(side, board, flags, depth, breadth, chooser)` takes a `chooser`
callable. The tests pass a `nonrandomChoice` that cycles deterministically
instead of sampling, so **always call `chooser(list)` rather than
`random.choice(list)`** or the results will not reproduce.

## Running

```bash
python grade.py                 # also --gradescope, and --profiler for timings
python main.py --player0 human --player1 alphabeta --depth1 3
python main.py --player0 minimax --player1 stochastic --breadth1 3 --loadgame res/savedGames/game0.txt
```
`grade.py --profiler` exists because these searches are slow; use it before
assuming a test hangs. Needs pygame (the tests set `PYGAME_HIDE_SUPPORT_PROMPT`).

Weights are 7-8 points per position; `mp06_extracredit.zip` is a separate,
ungraded-by-default bundle.

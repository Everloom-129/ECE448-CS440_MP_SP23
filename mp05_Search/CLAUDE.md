# CLAUDE.md — mp05_Search

BFS and A* over mazes. **This MP grades differently from every other one** —
read the first section before changing anything.

## `grade.py` here is hand-rolled

There is **no `tests/` directory**. `grade.py` imports `submitted` directly,
builds `maze.Maze` objects from `data/part-N/<case>`, and scores three things
per maze:

1. `maze.validate_path(path)` returns `None` — the path is legal
2. `len(path)` equals the reference length — it is *optimal*, not merely valid
3. `maze.states_explored < 1.1 * reference` — **efficiency is graded**

That third check is the one that bites. A correct BFS that re-expands nodes, or
an A* with a weak heuristic, passes (1) and (2) and still loses points. The
`Maze` object counts every `__getitem__`, so exploring is not free.

Run it with `python grade.py` (or `--gradescope`, **not** `-j` — this MP's flag
differs from the others).

## The answer key

`load_answer_key` looks for `key_i` (instructor) and falls back to `key_s`
(student). Only `key_s` is present, so you will see
`running in student mode (instructor key unavailable)` — that is expected, not
an error. `key_s` is a pickle of `(path_length, states_explored)` per case.

## The functions

* `bfs(maze)` — part 1
* `astar_single(maze)` — part 2, one waypoint
* `astar_multiple(maze)` — part 3, graded as **extra credit** with looser
  bounds (within 1.2x on both length and states explored)
* `manhattan_distance(a, b)` — the heuristic helper

All return a list of `(row, col)` tuples including start and goal.
`maze.start`, `maze.waypoints`, `maze.neighbors(i, j)` are the API to use.

## Seeing it run

```bash
python main.py data/part-1/tiny --search bfs      # also astar_single, astar_multiple
python main.py data/part-1/tiny --human           # play it yourself
python main.py data/part-2/small --search astar_single --save out.png
```
Needs pygame. `GIF/` holds recordings of bfs and A* for the writeup.

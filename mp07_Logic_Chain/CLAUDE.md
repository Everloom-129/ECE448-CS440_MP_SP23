# CLAUDE.md — mp07_Logic_Chain

Unification and backward chaining over a tiny first-order logic. Pure Python,
no numpy in the graded path. See the repo-root `CLAUDE.md` for conventions.

## Four functions, 12.5 points each

| function | returns |
|---|---|
| `standardize_variables(nonstandard_rules)` | `(standardized_rules, variables)` |
| `unify(query, datum, variables)` | `(unification, subs)` or `(None, None)` |
| `apply(rule, goals, variables)` | `(applications, goalsets)` |
| `backward_chain(query, rules, variables)` | a proof, or `None` |

## Representations you have to match exactly

* A **proposition** is a list `[subject, verb, object, truth_value]` where
  `truth_value` is a bool. Not a tuple, not a dict.
* A **rule** is `{'antecedents': [prop, ...], 'consequent': prop}`.
* `standardize_variables` replaces every occurrence of the literal string
  `"something"` with a fresh variable name **unique to that rule**. The
  reference uses names like `x0001`; the tests check that the names are
  distinct across rules and collected into `variables`, not that they match a
  particular spelling.
* `unify` must **not mutate** `query` or `datum` — deep-copy first. The
  docstring says so explicitly and the tests reuse the same objects.
* Variables can unify with variables; the substitution has to be applied
  transitively, so `unify(['x','eats','y',True], ['a','eats','b',True], ...)`
  chains.

## Data

`data/sample_data.jsonl` and `data/meta-train.jsonl` hold rule sets; `reader.py`
loads them. The tests build their own small rule sets inline, so you can debug
`unify` without touching the data files at all.

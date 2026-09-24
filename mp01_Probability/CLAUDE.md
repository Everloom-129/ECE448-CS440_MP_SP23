# CLAUDE.md — mp01_Probability

Joint, marginal and conditional distributions of word counts, estimated from a
directory of Enron emails. The repo-root `CLAUDE.md` covers the conventions
shared by every MP; this file covers what is specific here.

## The functions and their shapes

`submitted.py` is seven pure functions over numpy arrays — no classes, no state:

| function | returns |
|---|---|
| `joint_distribution_of_word_counts(texts, word0, word1)` | `Pjoint[m,n] = P(X0=m, X1=n)` |
| `marginal_distribution_of_word_counts(Pjoint, index)` | 1-D marginal over axis `index` |
| `conditional_distribution_of_word_counts(Pjoint, Pmarginal)` | `Pcond[m,n] = P(X1=n | X0=m)` |
| `mean_from_distribution(P)` / `variance_from_distribution(P)` | scalars |
| `covariance_from_distribution(P)` | scalar, from a 2-D `P` |
| `expectation_of_a_function(P, f)` | `E[f(X0,X1)]` for a two-argument `f` |

## The one thing that trips people up

**`Pjoint`'s shape is data-dependent and the test only checks a lower bound.**
`tests/test_visible.py` asserts `ref.shape[0] <= hyp.shape[0]` and then compares
the overlapping region to 2 decimal places, so a table that is *larger* than the
reference passes. It must be at least `max_count + 1` on each axis — the counts
are 0-inclusive. Sizing the array requires either two passes over `texts` (one
to find the maxima, one to fill) or growing it as you go; `submitted.py` does
the two-pass version and says so in a comment.

Rows of `Pcond` where the marginal is zero are `nan` in the reference, and the
test skips them — do not "fix" them to zero.

## Data and reference

* `reader.loadDir('data', stemming, lower_case)` returns `(texts, count)` where
  each text is a list of words. Both flags are `False` in the notebook.
* `solution.json` holds `Pjoint`, `P0`, `P1`, `Pcond`, and the scalar answers.
  Every test loads it, so later functions are graded on the *reference* inputs,
  not on your own earlier output — a wrong `Pjoint` does not cascade.
* `data/` is ~1500 plain-text emails; loading takes a few seconds.

Weights: joint 9, marginal 9, conditional 8, then 8 each for the moments.

# CLAUDE.md — mp08_Hidden_Markov_Model

Part-of-speech tagging: a baseline tagger and a Viterbi HMM, graded on
**accuracy thresholds** rather than against a stored reference. See the
repo-root `CLAUDE.md` for shared conventions.

## Graded on accuracy, with three separate bars

`tests/test_visible.py` runs each tagger over the Brown corpus and requires
**all three** of these at once:

| metric | minimum |
|---|---|
| overall accuracy | 0.935 |
| multi-tag word accuracy | 0.900 |
| **unseen word accuracy** | **0.670** |

The unseen-word bar is the hard one and it is what the smoothing is for. A
tagger that simply assigns the most common tag to unknown words will clear the
first two and fail the third. `utils.specialword_accuracies` computes them.

`tests/test_visible_ec.py` is auto-discovered alongside the visible tests
(`discover` matches `test*.py`) and grades `viterbi_ec` for extra credit, with
partial credit rather than a pass/fail bar.

There is also a `test_synthetic` sanity check that prints a warning band at
accuracy < 0.3, 0.3-0.9, and >= 0.9 — useful for a fast signal before the slow
Brown run.

## The functions

`baseline(train, test)` and `viterbi(train, test)` are the graded entry points;
both take and return lists of sentences, where a tagged sentence is a list of
`(word, tag)` pairs and a test sentence is a list of bare words.

The Viterbi path is factored into helpers you can unit-test separately:
`compute_initial_likelihoods`, `compute_transition_likelihoods`,
`compute_emission_likelihoods` (each takes an `alpha` smoothing constant), and
`trellis_backtrace(words, initial_p, transition_p, emission_p)`.
`improved_emission` is the extra-credit variant with a tag-dependent alpha.

## Practical notes

* **Work in log space.** Sentence probabilities underflow float64 well before
  the end of a Brown sentence.
* `data/brown-training.txt` and `data/brown-test.txt` are the graded pair;
  `synthetic_training.txt` / `synthetic_dev.txt` are the fast sanity set.
* `utils.load_dataset`, `utils.strip_tags` and `utils.get_word_tag_statistics`
  do the parsing — do not re-tokenize by hand.
* The Brown run takes a while; the test reports `time_spend` alongside accuracy.

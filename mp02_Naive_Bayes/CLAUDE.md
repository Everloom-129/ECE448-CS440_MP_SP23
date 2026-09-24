# CLAUDE.md — mp02_Naive_Bayes

Naive Bayes sentiment classification over movie reviews, built bottom-up from
raw frequency counts. See the repo-root `CLAUDE.md` for shared conventions.

## The pipeline

Each function consumes the previous one's output, and each is graded against
the *reference* input from `solution.json`, so a mistake early does not cascade:

1. `create_frequency_table(train)` → `frequency[y][x]` = count of word `x` in class `y`
2. `remove_stopwords(frequency)` → same shape, stopwords dropped
3. `laplace_smoothing(nonstop, smoothness)` → `likelihood[y][x]` = P(x|y), with
   an `OOV` key for unseen words
4. `naive_bayes(texts, likelihood, prior)` → predicted labels
5. `optimize_hyperparameters(...)` → grid search over priors × smoothnesses

## Things that are not obvious

* **The stopword list is hard-coded at the top of `submitted.py`**, not imported
  from nltk. It is a literal `set(...)` and it deliberately contains fragments
  like `"'t"`, `"'s"`, `"'ve"` because the tokenizer splits contractions.
  Do not swap in `nltk.corpus.stopwords` — the reference counts will not match.
* **`remove_stopwords` must not mutate its argument.** Build a fresh
  `Counter` per class rather than deleting from the caller's dict — the code
  carries a comment about this because it bit the author. The unit tests happen
  to survive it (`setUp` reloads `solution.json` before every test), but the
  notebook reuses one `frequency` across cells, so an in-place delete silently
  corrupts everything downstream of the first call.
* **Every likelihood needs an `OOV` entry.** `naive_bayes` is graded on texts
  containing words absent from training; missing `OOV` is a `KeyError`, not a
  wrong answer.
* Work in log space in `naive_bayes` — the products underflow on real reviews.

## Data

`reader.loadTrain(dirname, stemming, lower_case)` and `reader.loadDev(...)`;
`data/train` and `data/dev` hold `pos/` and `neg/` subdirectories.
Counts are compared with `assertEqual` (exact), likelihoods to 4 places.
Weights: 10 each for frequency / nonstop / likelihood, then the classifier.

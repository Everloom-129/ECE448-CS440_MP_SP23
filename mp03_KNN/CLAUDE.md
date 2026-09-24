# CLAUDE.md — mp03_KNN

k-nearest-neighbours image classification. Three functions, no training loop.
See the repo-root `CLAUDE.md` for shared conventions.

## The contract

| function | returns |
|---|---|
| `k_nearest_neighbors(image, train_images, train_labels, k)` | `(neighbors, labels)` — the k *images* and their labels |
| `classify_devset(dev_images, train_images, train_labels, k)` | `(hypotheses, scores)` — label and the winning vote count per dev image |
| `confusion_matrix(hypotheses, references)` | `(confusions, accuracy, f1)` |

`k_nearest_neighbors` returns the neighbour **images**, not their indices — the
test compares `neighbors` elementwise against a reference array of pixels.

`classify_devset` returns two parallel lists: the majority label *and* the
number of votes it got. Ties in the vote are broken in favour of the label the
reference implementation picks; `reference.py` is shipped in this directory and
is the fastest way to check a disagreement.

`confusion_matrix` returns `confusions[reference][hypothesis]` (true class as
the row). `f1` is for the positive class, not macro-averaged.

## Practical notes

* **No `requirements.txt` here.** It needs numpy only (plus `gradescope-utils`
  for the tests); `reader.py` uses `pickle`.
* Data lives in `mp3_data`, loaded by `reader.load_dataset`. It is small enough
  that a brute-force `np.linalg.norm` over all training images per query is
  fine — `submitted.py` does exactly that and sorts with `np.argsort`.
* Everything is graded against `solution.json` to 2 decimal places, and each
  test feeds the reference inputs, so errors do not cascade.
  Weights: 20 / 15 / 15.

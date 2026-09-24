# CLAUDE.md — mp04_NN_with_Pytorch

A small PyTorch classifier trained from scratch. The first MP where the grade
depends on a *trained* model rather than a closed-form answer. See the
repo-root `CLAUDE.md` for shared conventions.

## What is graded, and how

`tests/test_visible.py` calls `submitted.fit(train_loader, test_loader, epochs)`
and expects `(model, losses, ...)` back, then scores three things:

| test | weight | check |
|---|---|---|
| `test_loss_fn` | 10 | the returned loss is a `torch.nn.modules.loss._Loss` |
| `test_optimizer` | 15 | the returned optimizer is a `torch.optim.Optimizer` |
| `test_accuracy` | 40 (partial) | accuracy thresholds, **plus a parameter-count band** |

**The parameter count is a hard gate in both directions.** The model must have
strictly between **10,000 and 1,000,000** parameters. Too few fails with
"suspiciously compact"; too many fails outright. Check with
`sum(np.prod(w.shape) for w in model.parameters())` before submitting.

Accuracy is scored in +5 steps at **0.15 / 0.25 / 0.48 / 0.55** and stops at
the first threshold missed, so 0.47 scores the same as 0.26. Aim past 0.55.

## Practical notes

* The test passes `--epochs` (default 50) through argparse; `fit` must honour
  the `epochs` argument rather than hard-coding a loop length.
* `model(self.test_set)` is called directly on a tensor, so `forward` has to
  accept a batch of the raw test tensor — not a DataLoader.
* Two dependency files ship here: `requirements.txt` (pip) and
  `environment.yml` (conda, python 3.9). Either works; the conda one pins the
  versions the course used.
* `pytorch_tutorial.ipynb` is staff-provided background, not graded.
* Training on CPU is the norm for this MP; the dataset in `data/mp_data` is
  small.

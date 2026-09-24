# CLAUDE.md — mp09_Perception

CIFAR-10 classification: build the Dataset and DataLoader yourself, then
fine-tune a provided ResNet-18. See the repo-root `CLAUDE.md` for conventions.

## What the visible tests actually check

Only the data plumbing is covered by the visible tests, 15 points each:

* `test_dataset` — `build_dataset` yields **exactly 8000** test items
* `test_dataloader` — `build_dataloader` produces the expected batch count

The model, optimizer and accuracy are graded by hidden tests on Gradescope, so
a green local run does **not** mean the MP is done.

## The pieces

| function | note |
|---|---|
| `unpickle(file)` | CIFAR batches are pickled with `encoding='bytes'` — keys are byte strings like `b'data'`, not `'data'` |
| `CIFAR10(Dataset)` | must implement `__len__` and `__getitem__`; images are stored flat and need reshaping to 3x32x32 |
| `get_preprocess_transform(mode)` | separate train/test transforms |
| `build_model(trained=False)` | wraps `models.resnet18`; `trained=True` loads `resnet18.pt` |
| `build_optimizer(optim_type, model_params, hparams)` | dispatch on a string name |
| `train` / `test` | standard loops |

## Two file-layout facts

* `models.py` is **staff-provided** and defines `resnet18`; `resnet18.pt` next
  to it holds the pretrained weights. `submitted.py` does
  `from models import resnet18` — keep that import working.
* `cifar10_batches/` holds the raw `data_batch_1..5` and `test_batch` files.
  `build_dataset` takes a list of these paths.

Torch and torchvision are both required here (see `requirements.txt`); this is
the heaviest MP to set up.

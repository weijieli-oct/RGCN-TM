# differentRe: Cylinder Flow Across Different Reynolds Numbers

## Setup

Run all commands from this directory.

```bash
python -m pip install torch "numpy<2" matplotlib Pillow openpyxl
```

For GPU use, install a CUDA-enabled PyTorch build. For CPU use, set `CONFIG["device"] = "cpu"` in `params.py` and `DEVICE = "cpu"` in `predict_unseen.py`.

## Training

Edit the configuration in `params.py`:

```python
TRAIN_OUTPUT = "runs/refined_train"
RESUME_CHECKPOINT = None
PLOT_AFTER_TRAIN = True
```

```bash
python train.py
```

No external training dataset is required. Relative paths are resolved from this code directory.

To resume, set `RESUME_CHECKPOINT` to an existing `checkpoint.pt` and run the same command. Training settings are restored from the checkpoint. Resumed results are appended to the checkpoint directory.

## Plot Saved Results

In `predict.py`, leave `TRAIN_RESULT_DIRECTORY = None` to use the latest training run, or set it to an existing result directory.

```bash
python predict.py
```

Plots are saved under `plots_existing/` in the result directory.

## Run Unseen Cases

After training, edit `predict_unseen.py`:

```python
PREDICT_CHECKPOINT = None
PREDICT_RE = [85.0, 105.0, 125.0, 145.0, 75.0, 155.0]
PREDICT_OUTPUT = "runs/refined_unseen_corrected"
REPLOT_DIRECTORY = None
```

`PREDICT_CHECKPOINT = None` selects the latest training checkpoint. Choose cases absent from that checkpoint's training set.

```bash
python predict_unseen.py
```

This run uses physics correction. To redraw saved results, set `REPLOT_DIRECTORY` to an existing prediction directory and run the same command.

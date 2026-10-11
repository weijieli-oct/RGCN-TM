# differentGeo: Flow Across Different Geometries

## Setup

Run all commands from this directory.

```bash
python -m pip install torch "numpy<2" matplotlib Pillow openpyxl
```

For GPU use, install a CUDA-enabled PyTorch build. For CPU use, set `CONFIG["device"] = "cpu"` in `params.py` and `DEVICE = "cpu"` in `predict_unseen.py`.

## Training

Edit the configuration in `params.py`:

```python
TRAIN_OUTPUT = "runs/geometry_train"
RESUME_CHECKPOINT = None
PLOT_AFTER_TRAIN = True
```

```bash
python train.py
```

No external training dataset is required. Relative paths are resolved from this code directory.

To resume, set `RESUME_CHECKPOINT` to an existing `checkpoint.pt` and run the same command. Training settings are restored from the checkpoint. Resumed results are saved in a new directory.

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
PREDICT_SIDES = [13, 16, 20, 32]
PREDICT_OUTPUT = "runs/geometry_unseen_corrected"
REPLOT_DIRECTORY = None
```

`PREDICT_CHECKPOINT = None` selects the latest training checkpoint. Choose cases absent from that checkpoint's training set.

```bash
python predict_unseen.py
```

This run uses physics correction. To redraw saved results, set `REPLOT_DIRECTORY` to an existing prediction directory and run the same command.

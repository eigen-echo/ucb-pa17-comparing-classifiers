# Running SVM on DGX Spark (GPU)

Notebooks `02-model-training-additional-full.ipynb` and `03-model-training-full.ipynb` are configured to use NVIDIA's RAPIDS `cuml.svm.SVC` when available, falling back to `sklearn.svm.SVC` automatically on CPU machines. These instructions cover the end-to-end setup and execution on a DGX Spark device.

For the generic Windows / CPU environment, see `requirements.txt` in the repo root and install with `pip install -r requirements.txt`.

---

## Step 1 - Verify GPU and CUDA driver

```bash
nvidia-smi
```

Note the **CUDA Version** shown in the top-right corner (e.g. `13.0`). You need this to select the correct RAPIDS build in Step 4.

Also confirm you are on ARM64:

```bash
uname -m    # should print: aarch64
```

---

## Step 2 - Check if RAPIDS is already installed

Activate any existing Python environment and run:

```bash
# Quick import check
python -c "import cuml; print(cuml.__version__)"

# If using miniforge / conda
conda list cuml

# If using pip
pip show cuml-cu12
```

- Version number printed → RAPIDS is installed. Skip to [Step 5](#step-5--verify-the-installation-end-to-end).
- `ModuleNotFoundError` → proceed to Step 3.

> **Note:** This project uses miniforge. The `conda list` and `mamba` commands work identically to standard conda - miniforge is fully compatible with the same CLI.

---

## Step 3 - Check for a pre-built RAPIDS container (fastest path)

NVIDIA ships RAPIDS containers validated for DGX Spark hardware. Check if one is already available locally:

```bash
# List local images
docker images | grep rapids

# Or pull directly from NGC
docker pull nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.11
```

Using the container avoids build and compatibility issues entirely. If your notebook workflow already runs inside a container, this is the recommended path.

---

## Step 4 - Set up the miniforge environment

If no container is available, follow these steps in order to create and fully configure the `bank-svm` environment.

> **Why a dedicated environment?** Without one, a system-wide `pip install --upgrade scikit-learn` - or any package that pulls in a newer sklearn as a dependency - silently breaks cuML compatibility for every project on the machine at once with no easy rollback. A named conda env scopes all changes to this project only. See the [Troubleshooting & notes](#troubleshooting--notes) section for a concrete example.

### 4a - Create the conda environment (RAPIDS via mamba)

RAPIDS (`cuml`, `cupy`, `cudf`) cannot be installed via pip on aarch64 - they must come from the RAPIDS conda channel. The command below installs RAPIDS and all core dependencies in one step. All pinned versions are documented in `requirements-dgx.txt` at the repo root.

```bash
mamba create -n bank-svm -c rapidsai -c conda-forge -c nvidia \
    rapids=25.02 python=3.11 'cuda-version=12.8' \
    'scikit-learn=1.5' imbalanced-learn jupyterlab \
    pandas numpy matplotlib seaborn papermill
```

> **CUDA version note:** `nvidia-smi` on the DGX Spark shows CUDA 13.0, but stable RAPIDS releases currently top out at CUDA 12.8. CUDA drivers are backward compatible, so pinning `cuda-version=12.8` works correctly on a 13.0 driver. Before running, check the [RAPIDS install selector](https://docs.rapids.ai/install) (**conda → Stable → aarch64**) - if CUDA 13.0 appears in the dropdown, use that instead.

> **scikit-learn version note:** `scikit-learn` is pinned to `1.5` above. sklearn 1.6 introduced `__sklearn_tags__()` which cuML's base class has not yet implemented - using 1.6+ causes an `AttributeError` when sklearn's `Pipeline` calls `check_is_fitted` on a cuML estimator.

### 4b - Activate the environment

```bash
conda activate bank-svm
```

Always activate this environment before running any training commands. Every command from this point onwards assumes it is active.

### 4c - Install supplementary pip packages

With the environment active, install any remaining pip-only packages using `requirements-dgx.txt`:

```bash
pip install -r requirements-dgx.txt
```

This is a lightweight step - most packages were already pulled in by the `mamba create` command. The file is kept in the repo so the full dependency picture is version-controlled and reproducible.

### 4d - Register the environment as a Jupyter kernel

Required only if you plan to use papermill or open the notebooks in Jupyter:

```bash
python -m ipykernel install --user --name bank-svm --display-name "Python (bank-svm)"
```

Confirm it registered:

```bash
jupyter kernelspec list
# Expected output includes:
#   bank-svm   ~/.local/share/jupyter/kernels/bank-svm
```

---

## Step 5 - Verify the installation end-to-end

With `bank-svm` active, run this in the Python REPL or a notebook cell:

```python
import cupy, cuml
from cuml.svm import SVC
import numpy as np

print("cuML version         :", cuml.__version__)
print("GPU count            :", cupy.cuda.runtime.getDeviceCount())

props = cupy.cuda.runtime.getDeviceProperties(0)
print("Device name          :", props["name"].decode())
print("Streaming Multiprocs :", props["multiProcessorCount"])
print("Total VRAM           :", round(props["totalGlobalMem"] / 1024**3, 1), "GB")
print("Compute capability   :", str(props["major"]) + "." + str(props["minor"]))

# Smoke test
X = np.random.randn(200, 5).astype("float32")
y = (X[:, 0] > 0).astype("int32")
SVC(kernel="rbf").fit(X, y)
print("SVC smoke test       : OK")
```

Expected output on DGX Spark:

```
cuML version         : 25.02.xx
GPU count            : 1
Device name          : NVIDIA GB10
Streaming Multiprocs : 48
Total VRAM           : 119.7 GB
Compute capability   : 12.1
SVC smoke test       : OK
```

> **GPU count = 1 is correct.** The GB10 is a single GPU die with 48 internal Streaming Multiprocessors (6,144 CUDA cores). cuML dispatches CUDA kernels across all 48 SMs automatically inside each fit - no multi-device configuration is needed.

If `SVC smoke test` throws a CUDA kernel error, your RAPIDS build may not include `sm_121` support. Check for a newer RAPIDS release or nightly build at [https://docs.rapids.ai/install](https://docs.rapids.ai/install).

---

## Step 6 - Download the dataset

From the repo root with `bank-svm` active:

```bash
python src/setup.py
```

This downloads and extracts all four dataset variants into the `data/` folder. If the UCI site is down, see the fallback options in the main README.

---

## Step 7 - Confirm the GPU backend is active

The first cell of notebooks 02 and 03 (and the top of `src/train_additional_full.py`) prints which backend was loaded:

```
SVC backend: cuml     ← GPU path (DGX Spark)
SVC backend: sklearn  ← CPU fallback (laptop / bank-svm env not active)
```

If you see `sklearn` on the DGX Spark, the `bank-svm` environment is not active or cuML was not installed correctly - re-check Steps 2–4.

---

## Step 8 - Run the training

There are two ways to run training from the command line. Choose whichever fits your workflow.

### Option A - Python script (recommended for headless runs)

A standalone script `src/train_additional_full.py` mirrors notebook 02 exactly. It uses Python's `logging` module to stream timestamped output to both the terminal and a log file, and saves all plots as PNGs - no display or Jupyter kernel required.

```bash
conda activate bank-svm
python src/train_additional_full.py
```

To capture terminal output to a file as well:

```bash
python src/train_additional_full.py 2>&1 | tee outputs/02/terminal.log
```

Outputs are written to `outputs/02/`:

| File | Contents |
|------|----------|
| `run.log` | Full timestamped log of every section, metric, and GridSearchCV fit |
| `lr_confusion_roc.png` | Logistic Regression confusion matrix + ROC curve |
| `knn_confusion_roc.png` | KNN confusion matrix + ROC curve |
| `dt_confusion_roc.png` | Decision Tree confusion matrix + ROC curve |
| `svm_confusion_roc.png` | SVM confusion matrix + ROC curve |
| `comparison_performance_gstime.png` | Accuracy / ROC-AUC / GridSearch time bar charts |
| `comparison_train_predict_time.png` | Training vs prediction time bar charts |
| `comparison_roc_curves.png` | Overlaid ROC curves for all models |
| `lr_coefficients.png` | Logistic Regression coefficient plot |

What the live output looks like:

```
2026-04-10 14:02:01  INFO     SVC backend: cuml
2026-04-10 14:02:01  INFO     ============================================================
2026-04-10 14:02:01  INFO       Loading data
2026-04-10 14:02:01  INFO     ============================================================
2026-04-10 14:02:03  INFO     Shape: (41188, 20)
...
2026-04-10 14:08:45  INFO     Tuning SVM  (40 combinations × 5 folds = 200 fits) ...
[CV 1/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  4.2s
[CV 2/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  3.9s
...
2026-04-10 14:14:12  INFO       Best params  : {'model__C': 10, 'model__kernel': 'rbf', ...}
2026-04-10 14:14:12  INFO       CV ROC-AUC   : 0.8041
2026-04-10 14:14:12  INFO     Done. All outputs written to: .../outputs/02
```

---

### Option B - Notebook via papermill

Executes the notebook in place and streams each cell's output live - useful when you want the `.ipynb` output cells populated for later review in Jupyter.

```bash
conda activate bank-svm

# Run notebook 02 only
papermill notebooks/02-model-training-additional-full.ipynb \
          notebooks/02-model-training-additional-full.ipynb \
          --kernel bank-svm \
          --log-output \
          --log-level INFO \
          --progress-bar 2>&1 | tee notebooks/02-model-training-additional-full.log

# Or run both notebooks sequentially
for nb in notebooks/02-model-training-additional-full.ipynb \
           notebooks/03-model-training-full.ipynb; do
    echo "======== Running $nb ========"
    papermill "$nb" "$nb" \
        --kernel bank-svm \
        --log-output \
        --log-level INFO \
        --progress-bar 2>&1 | tee "${nb%.ipynb}.log"
done
```

What the live output looks like:

```
Executing:  18%|████▌      | 6/34 [00:12<00:48]   ← papermill cell progress bar
...
Tuning SVM...
Fitting 5 folds for each of 40 candidates, totalling 200 fits
[CV 1/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  4.2s
[CV 2/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  3.9s
...
[CV 5/5] END model__C=100, model__kernel=rbf,    model__gamma=auto;  total time=  8.7s
  Best params  : {'model__C': 10, 'model__kernel': 'rbf', 'model__gamma': 'scale'}
  GridSearch   : 312.4s
```

---

### Which option to use

| | Python script | papermill |
|---|---|---|
| Jupyter / kernel required | No | Yes (`--kernel bank-svm`) |
| Output cells saved in `.ipynb` | No | Yes |
| Plots | PNG files in `outputs/02/` | Embedded in notebook |
| Logging | Timestamped via `logging` module | Cell stdout streamed live |
| Best for | Headless server runs, automation | Keeping notebook outputs for review |

> `verbose=2` on `GridSearchCV` is only active when `_SVC_BACKEND == 'cuml'` (i.e. on the DGX Spark). On a CPU laptop the SVM grid search stays silent in both modes.

---

## Monitoring GPU utilisation

Open a second terminal while the script or papermill is running:

```bash
watch -n 1 nvidia-smi
```

The Blackwell GPU should show non-trivial utilisation and VRAM usage during the SVM block, and drop to idle during Logistic Regression, KNN, and Decision Tree fits (those remain on CPU).

---

## Troubleshooting & notes

### `AttributeError: __sklearn_tags__` when running SVM

**Symptom:** Training runs fine through LR / KNN / Decision Tree, then fails at the SVM section with:

```
AttributeError: __sklearn_tags__
  File "sklearn/pipeline.py", line 1254, in __sklearn_is_fitted__
  File "cuml/internals/base.pyx", line 398, in cuml.internals.base.Base.__getattr__
```

**Root cause:** scikit-learn 1.6 introduced a new `__sklearn_tags__()` method for estimator introspection. cuML's base class has not yet implemented it. When sklearn's `Pipeline` calls `check_is_fitted` on the cuML SVC estimator after fitting, it tries to call this method and fails.

**Fix:** Downgrade scikit-learn to 1.5.x in the active environment:

```bash
mamba install -n bank-svm "scikit-learn=1.5"
```

The `mamba create` command in Step 4a and the `requirements-dgx.txt` file both already pin `scikit-learn=1.5` to prevent this on a fresh install.

---

### Why you should always use a virtual environment

The bug above is a concrete example of why isolated environments matter. Without one, a system-wide `pip install --upgrade scikit-learn` - or any package that pulls in a newer sklearn as a transitive dependency - silently breaks cuML compatibility for every project on the machine at once, with no easy way to roll back.

With a named conda environment (`bank-svm`), the fix is one command scoped only to this project, and every other environment on the machine is unaffected.

| Problem | Without env | With env |
|---|---|---|
| Package A needs sklearn 1.5, Package B needs 1.6 | One of them breaks | Each gets its own version |
| Upgrading a system package breaks your project | Hard to diagnose, hard to undo | Isolated - system changes don't affect it |
| Reproducing results on another machine | "Works on my machine" | `mamba env export` captures exact versions |
| Sharing the project with a colleague | Manual dependency hunting | Activate env, run script |

---

## Quick decision tree

```
nvidia-smi shows GPU?
  └─ No  → hardware/driver issue, contact NVIDIA support
  └─ Yes → python -c "import cuml" works?
              └─ Yes → conda activate bank-svm, then Step 8
              └─ No  → docker images | grep rapids has a container?
                          └─ Yes → use the container
                          └─ No  → follow Steps 4a–4d (mamba create + pip install -r requirements-dgx.txt)
```

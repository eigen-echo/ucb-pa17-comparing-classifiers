# Running SVM on DGX Spark (GPU)

Notebooks `02-model-training-additional-full.ipynb` and `03-model-training-full.ipynb` are configured to use NVIDIA's RAPIDS `cuml.svm.SVC` when available, falling back to `sklearn.svm.SVC` automatically on CPU machines. These instructions cover how to set up and verify the GPU environment on a DGX Spark device.

---

## Step 1 — Verify GPU and CUDA driver

```bash
nvidia-smi
```

Note the **CUDA Version** shown in the top-right corner of the output (e.g. `12.8`). You need this to select the correct RAPIDS build.

---

## Step 2 — Check if RAPIDS is already installed

Activate the Python environment you intend to use for the notebooks, then run any of the following:

```bash
# Quick import check
python -c "import cuml; print(cuml.__version__)"

# If using miniforge / conda, list the cuML package
conda list cuml

# If using pip
pip show cuml-cu12
```

- If a version number is printed → RAPIDS is installed. Skip to [Step 5](#step-5--verify-the-installation-end-to-end).
- If you get `ModuleNotFoundError` → proceed to Step 3.

> **Note:** This project uses miniforge. The `conda list` and `mamba` commands work identically to standard conda — miniforge is fully compatible with the same CLI.

---

## Step 3 — Check for a pre-built RAPIDS container (fastest path)

NVIDIA ships RAPIDS containers validated for DGX Spark hardware. Check if one is already available locally before installing from scratch:

```bash
# List local images
docker images | grep rapids

# Or pull directly from NGC
docker pull nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.11
```

Using the container avoids build and compatibility issues entirely. If your notebook workflow already runs inside a container, this is the recommended path.

---

## Step 4 — Install RAPIDS into a miniforge environment

If a container is not available, create a fresh conda environment using `mamba` (ships with miniforge and is significantly faster than `conda` for dependency solving):

```bash
# 1. Confirm you are on ARM64 (expected on DGX Spark)
uname -m    # should print: aarch64

# 2. Create the environment (substitute the CUDA version from nvidia-smi)
mamba create -n bank-svm -c rapidsai -c conda-forge -c nvidia \
    rapids=25.02 python=3.11 'cuda-version>=12.5' \
    scikit-learn imbalanced-learn jupyterlab \
    pandas numpy matplotlib seaborn papermill

# 3. Activate
conda activate bank-svm
```

> Substitute `25.02` with the latest RAPIDS release that matches your CUDA version. Use the [RAPIDS install selector](https://docs.rapids.ai/install) and choose: **conda**, **Stable**, **aarch64**, and your CUDA version.

---

## Step 5 — Verify the installation end-to-end

Run this in a notebook cell or directly in the Python REPL:

```python
import cupy, cuml
from cuml.svm import SVC

print("cuML version  :", cuml.__version__)
print("GPU count     :", cupy.cuda.runtime.getDeviceCount())

# Smoke test: tiny fit
import numpy as np
X = np.random.randn(100, 5).astype("float32")
y = (X[:, 0] > 0).astype("int32")
svc = SVC(kernel="rbf")
svc.fit(X, y)
print("SVC fit OK    :", svc.predict(X[:3]))
```

If all lines print without error the environment is ready to run the notebooks.

---

## Step 6 — Confirm the backend when running the notebooks

The first cell of notebooks 02 and 03 prints which backend was loaded:

```
SVC backend: cuml     ← GPU path (DGX Spark)
SVC backend: sklearn  ← CPU fallback (laptop)
```

If you see `sklearn` on the DGX Spark, the `bank-svm` environment is not active or cuML was not installed correctly — re-check Steps 2–4.

---

## Step 7 — Run notebooks from the command line with papermill

`papermill` is the recommended way to execute notebooks on the Spark. Unlike `jupyter nbconvert` (which buffers all output until the cell finishes), papermill streams each cell's `print()` output to the terminal in real time — essential for watching the 200-fit GridSearchCV progress.

### Install (already included in the Step 4 `mamba create` command)

```bash
pip install papermill   # only needed if you skipped the mamba create step
```

### Run a single notebook

```bash
papermill \
    notebooks/02-model-training-additional-full.ipynb \
    notebooks/02-model-training-additional-full.ipynb \
    --log-output \
    --log-level INFO \
    --progress-bar
```

- Both paths identical → outputs are saved back **in place**
- `--log-output` — streams `print()` from every cell live to the terminal
- `--log-level INFO` — also shows papermill's own cell-start/end events
- `--progress-bar` — top-level bar showing which cell number is executing

### Run both notebooks sequentially, saving a log file alongside each

```bash
for nb in notebooks/02-model-training-additional-full.ipynb \
           notebooks/03-model-training-full.ipynb; do
    echo "======== Running $nb ========"
    papermill "$nb" "$nb" \
        --log-output \
        --log-level INFO \
        --progress-bar 2>&1 | tee "${nb%.ipynb}.log"
done
```

Each notebook gets a `.log` file next to it (e.g. `02-model-training-additional-full.log`) with the full run trace.

### What the live output looks like

Because `verbose=2` is set on `GridSearchCV` for the cuML/GPU code path, you will see every fit streamed as it completes:

```
Executing:  18%|████▌      | 6/34 [00:12<00:48]   ← papermill cell progress bar
...
Tuning SVM...
Fitting 5 folds for each of 40 candidates, totalling 200 fits
[CV 1/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  4.2s
[CV 2/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  3.9s
[CV 3/5] END model__C=0.01, model__kernel=linear, model__gamma=scale; total time=  4.1s
...
[CV 5/5] END model__C=100, model__kernel=rbf,    model__gamma=auto;  total time=  8.7s
  Best params  : {'model__C': 10, 'model__kernel': 'rbf', 'model__gamma': 'scale'}
  CV ROC-AUC   : 0.8041
  GridSearch   : 312.4s
```

> `verbose=2` is only active when `_SVC_BACKEND == 'cuml'` (i.e. on the DGX Spark). On a CPU laptop the SVM grid search stays silent.

---

## Monitoring GPU utilisation during GridSearchCV

Open a second terminal while papermill is running:

```bash
watch -n 1 nvidia-smi
```

The Blackwell GPU should show non-trivial utilisation and VRAM usage during the SVM block, and drop to idle during Logistic Regression, KNN, and Decision Tree fits (those remain on CPU).

---

## Quick decision tree

```
nvidia-smi shows GPU?
  └─ No  → hardware/driver issue, contact NVIDIA support
  └─ Yes → python -c "import cuml" works?
              └─ Yes → run notebooks (expect "SVC backend: cuml")
              └─ No  → docker images | grep rapids has a container?
                          └─ Yes → use the container
                          └─ No  → run mamba create command in Step 4
```

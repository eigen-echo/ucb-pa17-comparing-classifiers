# Comparing Classifiers - Bank Marketing Campaign

UC Berkeley Professional Certificate in ML/AI · Practical Application 17

A comparison of four classifiers - **Logistic Regression**, **K-Nearest Neighbors**, **Decision Tree**, and **Support Vector Machine** - applied to a real-world bank telemarketing dataset to predict whether a client will subscribe to a term deposit.

---

## Project Directory

```
ucb-pa17-comparing-classifiers/
├── data/                            # Dataset files (populated by setup.py)
│   ├── bank.csv                     # Small dataset  (~4,500 rows, 15 features)
│   ├── bank-full.csv                # Full dataset   (~45,000 rows, 15 features)
│   ├── bank-additional.csv          # Small dataset  (~4,500 rows, 20 features)
│   ├── bank-additional-full.csv     # Full dataset   (~41,000 rows, 20 features)
│   └── bank-names.txt               # Feature descriptions
├── docs/
│   ├── assignment_instructions.md   # Original assignment brief
│   ├── findings.md                  # Summary of findings across all notebooks
│   └── dgx-spark-insructions.md    # GPU environment setup (feature/setup-for-dgx-spark branch)
├── notebooks/
│   ├── 00-exploratary-data-analysis.ipynb
│   ├── 01-model-training-small-dataset.ipynb
│   ├── 02-model-training-additional-full.ipynb
│   ├── 03-model-training-full.ipynb
│   └── 04-model-training-small-dataset-smote.ipynb
├── src/
│   ├── setup.py                     # Data download script
│   └── train_additional_full.py    # Standalone training script for notebook 02 (feature/setup-for-dgx-spark branch)
└── README.md
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (Anaconda recommended)
- [Conda](https://docs.conda.io/en/latest/) or `pip`

### 2. Create a Conda Environment

```bash
conda create -n classifiers python=3.11
conda activate classifiers
```

### 3. Install Dependencies

```bash
pip install -r requirements-dgx.txt # feature/setup-for-dgx-spark branch
```

### 4. Download the Data

Run the setup script from the **project root** to download and extract all dataset files into `data/`:

```bash
python src/setup.py
```

This fetches the Bank Marketing dataset directly from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and extracts the relevant CSV files.

### 5. Launch Jupyter

```bash
jupyter notebook notebooks/
```

Run the notebooks **in order** (00 → 04). The EDA notebook (00) is standalone; the modelling notebooks (01–04) each load their own dataset and are independent of each other.

---

## Notebooks

| # | Notebook | Dataset | Description |
|---|----------|---------|-------------|
| 00 | `00-exploratary-data-analysis.ipynb` | `bank-full.csv` | **Exploratory Data Analysis.** Examines class imbalance (~88.5% "no"), categorical feature distributions, campaign contact patterns (method, timing, call frequency), and call duration. Establishes that a naive majority classifier already hits ~88.5% accuracy - making recall on the minority class the real challenge. |
| 01 | `01-model-training-small-dataset.ipynb` | `bank.csv` (~4,500 rows) | **Baseline modelling on the small dataset.** Trains all four classifiers with a `StandardScaler + OneHotEncoder` pipeline, evaluates with 5-fold stratified CV, then runs `GridSearchCV` for hyperparameter tuning. Includes a Logistic Regression coefficient plot showing feature importance. Good starting point due to fast iteration times. |
| 02 | `02-model-training-additional-full.ipynb` | `bank-additional-full.csv` (~41,000 rows, 20 features) | **Modelling on the richer 20-feature dataset.** Adds five macroeconomic indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) not present in the 15-feature format. SVM is run with `max_iter=2000` and a sigmoid kernel to keep runtime manageable. GridSearchCV is included but SVM tuning is compute-constrained (see note below). |
| 03 | `03-model-training-full.ipynb` | `bank-full.csv` (~45,000 rows, 15 features) | **Modelling on the full 15-feature dataset.** Same pipeline structure as notebook 02 but using the original feature set. Allows direct comparison with notebook 01 (same features, 10× more data) to observe how scale affects each classifier. SVM again capped at `max_iter=2000`. |
| 04 | `04-model-training-small-dataset-smote.ipynb` | `bank.csv` → tested on `bank-full.csv` | **Completed with GenAI assistance for helping me learn SMOTE. SMOTE oversampling to address class imbalance.** Re-trains all four models on the small dataset using `ImbPipeline` with SMOTE to synthesise minority-class examples before each training fold. Compares baseline (no resampling) vs SMOTE via `GridSearchCV`, focusing on minority-class recall. Ends with a generalisation test against the full `bank-full.csv`. |

---

## Findings

See **[docs/findings.md](docs/findings.md)** for a detailed summary of results, key observations, and actionable insights across all notebooks.

---

## A Note on SVM and GridSearchCV Runtime

> **This exercise is incomplete with respect to SVM hyperparameter tuning on the larger datasets.**

SVM with an RBF or linear kernel and no iteration cap ran for **over 6 hours** on the full datasets without converging - the available hardware simply couldn't complete a full `GridSearchCV` sweep in reasonable time. As a workaround, SVM in notebooks 02 and 03 is constrained to `max_iter=2000` with a sigmoid kernel, which converges quickly but is not optimally tuned.

If you run `GridSearchCV` on any of the larger notebooks, **expect your processor fans to spin up noticeably** - cross-validated grid search across four models on 45,000 rows is genuinely CPU-intensive. The KNN and Decision Tree searches are the most tractable; LR is fast; SVM is the bottleneck if trained on full dataset.

### Machine This Was Executed On

| Property | Value |
|----------|-------|
| **OS** | Windows 10 (Build 26200) |
| **CPU** | Intel Core i9-12900HK (12th Gen, Alder Lake) |
| **Physical Cores** | 14 |
| **Logical Processors** | 20 |
| **RAM** | 63.7 GB |

Even on this machine, unconstrained SVM GridSearchCV on the full dataset was not practical without overnight runs.

---

## Exploring SVM on GPU Hardware (Out of Scope / Learning Exercise)

> **Note:** This section describes work that is outside the requirements of the assignment. It was pursued out of curiosity and for personal learning, with heavy use of AI assistance (Claude). It is tracked in the branch [`feature/setup-for-dgx-spark`](../../tree/feature/setup-for-dgx-spark) and intentionally kept off the main branch.

The SVM runtime constraint documented above prompted an exploration of whether GPU-accelerated hardware could make uncapped, fully-tuned SVM GridSearchCV feasible on the larger datasets.

### Hardware

The experiment uses an **NVIDIA DGX Spark** - a personal AI supercomputer built around the GB10 Grace Blackwell Superchip:

| Property | Value |
|----------|-------|
| **GPU** | NVIDIA GB10 (Blackwell) |
| **Streaming Multiprocessors** | 48 |
| **CUDA cores** | 6,144 |
| **Unified Memory** | 128 GB (CPU + GPU shared pool) |
| **Compute Capability** | 12.1 |
| **Architecture** | aarch64 (ARM64 Grace CPU) |

See **[docs/dgx-findings.md](docs/dgx-findings.md)** for the full side-by-side comparison with plots, timing tables, and analysis.

### Branch and setup instructions

| | |
|---|---|
| **Branch** | [`feature/setup-for-dgx-spark`](../../tree/feature/setup-for-dgx-spark) |
| **Findings** | [`docs/dgx-findings.md`](docs/dgx-findings.md) |
| **GPU setup guide** | `docs/dgx-spark-insructions.md` |
| **GPU dependencies** | `requirements-dgx.txt` |
| **Training script** | `src/train_additional_full.py` |

---

## Data Source

[UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.

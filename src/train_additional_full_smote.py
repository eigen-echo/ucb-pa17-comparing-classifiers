"""
SMOTE training script for bank-additional-full.csv dataset.
Mirrors src/train_additional_full.py but adds SMOTE oversampling and runs
a side-by-side comparison: Baseline (no resampling) vs SMOTE.

Usage:
    python src/train_additional_full_smote.py

Outputs:
    - Plots saved to outputs/02-smote/
    - Log file at outputs/02-smote/run.log
    - Models saved to models/02-smote/  (baseline_*.joblib + smote_*.joblib)
"""

import json
import logging
import os
import platform
import sys
import time
import warnings
from datetime import datetime

import joblib
import sklearn

warnings.filterwarnings("ignore")

# Use non-interactive backend - no display required (safe for DGX Spark / headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup - writes to both console and a log file
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "02-smote")
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_path = os.path.join(OUTPUT_DIR, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
log.info("Log file: %s", os.path.abspath(log_path))

# ---------------------------------------------------------------------------
# GPU / CPU backend detection
# ---------------------------------------------------------------------------
try:
    from cuml.svm import SVC
    _SVC_BACKEND = "cuml"
except ImportError:
    from sklearn.svm import SVC
    _SVC_BACKEND = "sklearn"
from sklearn.svm import LinearSVC  # noqa: F401 - kept for parity with notebook

log.info("SVC backend: %s", _SVC_BACKEND)

# ---------------------------------------------------------------------------
# Scikit-learn imports
# ---------------------------------------------------------------------------
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline   # baseline (no SMOTE)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# imbalanced-learn imports
# ---------------------------------------------------------------------------
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # SMOTE-aware pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bank-additional-full.csv")


def save_fig(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Plot saved: %s", path)


def section(title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


# ===========================================================================
# 1. Load and preprocess data
# ===========================================================================
section("Loading data")
log.info("Reading: %s", os.path.abspath(DATA_PATH))
df = pd.read_csv(DATA_PATH, sep=";")
log.info("Shape: %s", df.shape)

# Drop call duration (leaks target at inference time)
df = df.drop(columns=["duration"])

# Encode target
df["y"] = (df["y"] == "yes").astype(int)
log.info("Class distribution:\n%s", df["y"].value_counts().to_string())

# Convert pdays (999 = no prior contact) to a binary flag
df["p_contacted"] = (df["pdays"] != 999).astype(int)
df = df.drop(columns=["pdays"])

cat_cols = df.select_dtypes(include="object").columns.tolist()
log.info("Categorical columns: %s", cat_cols)

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

num_cols = X_train.select_dtypes(include="number").columns.tolist()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# sparse_output=False required for cuML SVC (dense input) and for SMOTE
# (SMOTE operates on arrays, not sparse matrices)
preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
)

# ===========================================================================
# 2. Dummy classifier - baseline
# ===========================================================================
section("Dummy Classifier (baseline)")

dummy_pipe = SkPipeline([
    ("preprocessor", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
])
dummy_pipe.fit(X_train, y_train)
y_pred_dummy = dummy_pipe.predict(X_test)
y_prob_dummy = dummy_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy     : %.4f", accuracy_score(y_test, y_pred_dummy))
log.info("ROC-AUC      : %.4f", roc_auc_score(y_test, y_prob_dummy))
log.info("Recall (yes) : %.4f", recall_score(y_test, y_pred_dummy))

# ===========================================================================
# 3. SMOTE effect on class distribution
# ===========================================================================
section("SMOTE effect on class distribution")

_pre_vis = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
)
X_train_enc = _pre_vis.fit_transform(X_train)
_, y_resampled = SMOTE(random_state=42).fit_resample(X_train_enc, y_train)

before = y_train.value_counts().sort_index()
after  = pd.Series(y_resampled).value_counts().sort_index()

log.info("Before SMOTE - No: %d  Yes: %d  (ratio %.1f:1)",
         before[0], before[1], before[0] / before[1])
log.info("After  SMOTE - No: %d  Yes: %d  (ratio %.1f:1)",
         after[0], after[1], after[0] / after[1])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, counts, title in zip(axes, [before, after], ["Before SMOTE", "After SMOTE"]):
    ax.bar(["No (0)", "Yes (1)"], counts.values, color=["steelblue", "crimson"])
    ax.set_title(f"Training Set - {title}")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, f"{v:,}", ha="center", fontweight="bold")
plt.suptitle("SMOTE Effect on Class Distribution", fontsize=13)
plt.tight_layout()
save_fig("smote_class_distribution")

# ===========================================================================
# 4. Logistic Regression with SMOTE
# ===========================================================================
section("Logistic Regression (SMOTE)")

lr_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
y_prob_lr  = lr_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy     : %.4f", accuracy_score(y_test, y_pred_lr))
log.info("ROC-AUC      : %.4f", roc_auc_score(y_test, y_prob_lr))
log.info("Recall (yes) : %.4f", recall_score(y_test, y_pred_lr))
log.info("\n%s", classification_report(y_test, y_pred_lr, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - LR (SMOTE)")
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
axes[1].plot(fpr, tpr, label=f"LR SMOTE (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - LR (SMOTE)"); axes[1].legend()
plt.tight_layout()
save_fig("lr_smote_confusion_roc")

log.info("Logistic Regression (SMOTE) - 5-Fold Stratified CV")
cv_acc_lr = cross_val_score(lr_pipe, X, y, cv=skf, scoring="accuracy")
cv_auc_lr = cross_val_score(lr_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_acc_lr.mean(), cv_acc_lr.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_auc_lr.mean(), cv_auc_lr.std())

# ===========================================================================
# 5. K-Nearest Neighbors with SMOTE
# ===========================================================================
section("K-Nearest Neighbors (SMOTE)")

knn_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", KNeighborsClassifier()),
])
knn_pipe.fit(X_train, y_train)
y_pred_knn = knn_pipe.predict(X_test)
y_prob_knn  = knn_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy     : %.4f", accuracy_score(y_test, y_pred_knn))
log.info("ROC-AUC      : %.4f", roc_auc_score(y_test, y_prob_knn))
log.info("Recall (yes) : %.4f", recall_score(y_test, y_pred_knn))
log.info("\n%s", classification_report(y_test, y_pred_knn, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - KNN (SMOTE)")
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
axes[1].plot(fpr_knn, tpr_knn, label=f"KNN SMOTE (AUC = {roc_auc_score(y_test, y_prob_knn):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - KNN (SMOTE)"); axes[1].legend()
plt.tight_layout()
save_fig("knn_smote_confusion_roc")

log.info("KNN (SMOTE) - 5-Fold Stratified CV")
cv_acc_knn = cross_val_score(knn_pipe, X, y, cv=skf, scoring="accuracy")
cv_auc_knn = cross_val_score(knn_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_acc_knn.mean(), cv_acc_knn.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_auc_knn.mean(), cv_auc_knn.std())

# ===========================================================================
# 6. Decision Tree with SMOTE
# ===========================================================================
section("Decision Tree (SMOTE)")

dt_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", DecisionTreeClassifier(random_state=42)),
])
dt_pipe.fit(X_train, y_train)
y_pred_dt = dt_pipe.predict(X_test)
y_prob_dt  = dt_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy     : %.4f", accuracy_score(y_test, y_pred_dt))
log.info("ROC-AUC      : %.4f", roc_auc_score(y_test, y_prob_dt))
log.info("Recall (yes) : %.4f", recall_score(y_test, y_pred_dt))
log.info("\n%s", classification_report(y_test, y_pred_dt, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - DT (SMOTE)")
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
axes[1].plot(fpr_dt, tpr_dt, label=f"DT SMOTE (AUC = {roc_auc_score(y_test, y_prob_dt):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - DT (SMOTE)"); axes[1].legend()
plt.tight_layout()
save_fig("dt_smote_confusion_roc")

log.info("Decision Tree (SMOTE) - 5-Fold Stratified CV")
cv_acc_dt = cross_val_score(dt_pipe, X, y, cv=skf, scoring="accuracy")
cv_auc_dt = cross_val_score(dt_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_acc_dt.mean(), cv_acc_dt.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_auc_dt.mean(), cv_auc_dt.std())

# ===========================================================================
# 7. Support Vector Machine with SMOTE
# ===========================================================================
section("Support Vector Machine (SMOTE)")
log.info("SVC backend: %s", _SVC_BACKEND)

svm_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", SVC(
        C=1.0,
        kernel="rbf",
        random_state=42,
        # no cap on GPU; restore safeguard on CPU
        max_iter=(-1 if _SVC_BACKEND == "cuml" else 2000),
    )),
])
svm_pipe.fit(X_train, y_train)
y_pred_svm = svm_pipe.predict(X_test)
y_score_svm = svm_pipe.decision_function(X_test)

log.info("Accuracy     : %.4f", accuracy_score(y_test, y_pred_svm))
log.info("ROC-AUC      : %.4f", roc_auc_score(y_test, y_score_svm))
log.info("Recall (yes) : %.4f", recall_score(y_test, y_pred_svm))
log.info("\n%s", classification_report(y_test, y_pred_svm, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - SVM (SMOTE)")
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
axes[1].plot(fpr_svm, tpr_svm, label=f"SVM SMOTE (AUC = {roc_auc_score(y_test, y_score_svm):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - SVM (SMOTE)"); axes[1].legend()
plt.tight_layout()
save_fig("svm_smote_confusion_roc")

log.info("SVM (SMOTE) - 5-Fold Stratified CV")
cv_acc_svm = cross_val_score(svm_pipe, X, y, cv=skf, scoring="accuracy")
cv_auc_svm = cross_val_score(svm_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_acc_svm.mean(), cv_acc_svm.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_auc_svm.mean(), cv_auc_svm.std())

# ===========================================================================
# 8. GridSearchCV - Baseline vs SMOTE (all models)
# ===========================================================================
section("GridSearchCV - Baseline vs SMOTE (all models)")

# --- Baseline pipelines (plain sklearn Pipeline, no SMOTE) ---
base_pipes = {
    "Logistic Regression": SkPipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "KNN": SkPipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier()),
    ]),
    "Decision Tree": SkPipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]),
    "SVM": SkPipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(
            C=1.0, kernel="rbf", random_state=42,
            max_iter=(-1 if _SVC_BACKEND == "cuml" else 2000),
        )),
    ]),
}

base_param_grids = {
    "Logistic Regression": {
        "model__C":      [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 11, 21, 29, 41],
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan"],
    },
    "Decision Tree": {
        "model__max_depth":         [3, 5, 10, None],
        "model__min_samples_split": [2, 10, 20],
        "model__criterion":         ["gini", "entropy"],
    },
    "SVM": {
        "model__C":      [0.01, 0.1, 0.5],
        "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "model__gamma":  ["scale", "auto"],
    },
}

# --- SMOTE pipelines (ImbPipeline with SMOTE step) ---
smote_pipes = {
    "Logistic Regression": lr_pipe,
    "KNN":                 knn_pipe,
    "Decision Tree":       dt_pipe,
    "SVM":                 svm_pipe,
}

# SMOTE grids extend the baseline grids with smote__k_neighbors tuning
smote_param_grids = {
    "Logistic Regression": {
        "smote__k_neighbors": [3, 5, 7, 9],
        "model__C":           [0.01, 0.1, 1, 10],
        "model__solver":      ["lbfgs", "liblinear"],
    },
    "KNN": {
        "smote__k_neighbors":  [3, 5, 7, 9],
        "model__n_neighbors":  [3, 5, 11, 21, 29, 41],
        "model__weights":      ["uniform", "distance"],
        "model__metric":       ["euclidean", "manhattan"],
    },
    "Decision Tree": {
        "smote__k_neighbors":       [3, 5, 7, 9],
        "model__max_depth":         [3, 5, 10, None],
        "model__min_samples_split": [2, 10, 20],
        "model__criterion":         ["gini", "entropy"],
    },
    "SVM": {
        "smote__k_neighbors": [9, 13, 17, 21],
        "model__C":           [0.01, 0.1, 0.5],
        "model__kernel":      ["linear", "poly", "rbf", "sigmoid"],
        "model__gamma":       ["scale", "auto"],
    },
}

# ---------------------------------------------------------------------------
# Parallelism strategy
#
# GPU path (cuML):
#   Each GridSearchCV worker runs: CPU preprocessing → CPU SMOTE → GPU SVC fit.
#   The CPU-heavy SMOTE step naturally staggers GPU submissions across workers,
#   so N concurrent workers do NOT all hit the GPU simultaneously.
#   The DGX Spark's 128 GB unified memory gives ample headroom for several
#   concurrent SVM fits (each needs ~100–300 MB, not GB).
#
#   _SVM_GPU_JOBS controls how many CV fits run in parallel for SVM:
#     4  - conservative; safe starting point
#     8  - recommended for Spark (watch `nvidia-smi` for >80% SM utilization)
#     12 - aggressive; only raise if GPU is still under-utilised at 8
#   Halving wall-clock time from the serialized baseline (n_jobs=1) is realistic
#   at n_jobs=4; diminishing returns kick in as GPU becomes the bottleneck.
#
#   LR / KNN / DT run purely on CPU - n_jobs=-1 uses all available cores.
#   No GPU contention risk for those models.
#
# CPU path (sklearn):
#   All models use n_jobs=-1 (full CPU parallelism, no GPU to worry about).
# ---------------------------------------------------------------------------
_SVM_GPU_JOBS = 16   # tune: raise to 12 if `nvidia-smi` SM util stays below 80%

per_model_n_jobs  = {"SVM": _SVM_GPU_JOBS if _SVC_BACKEND == "cuml" else -1}
per_model_verbose = {"SVM": 3 if _SVC_BACKEND == "cuml" else 0}

log.info("SVM GridSearchCV n_jobs: %s",
         _SVM_GPU_JOBS if _SVC_BACKEND == "cuml" else "-1 (all CPU cores)")


def run_gridsearch(pipes, param_grids, label):
    """Run GridSearchCV for each model and return a list of result dicts."""
    results = []
    for name, pipe in pipes.items():
        n_combinations = 1
        for v in param_grids[name].values():
            n_combinations *= len(v)
        total_fits = n_combinations * skf.n_splits
        log.info("[%s] Tuning %s  (%d combinations × %d folds = %d fits) ...",
                 label, name, n_combinations, skf.n_splits, total_fits)
        t_start = time.time()

        gs = GridSearchCV(
            pipe,
            param_grids[name],
            cv=skf,
            scoring="roc_auc",
            n_jobs=per_model_n_jobs.get(name, -1),
            verbose=per_model_verbose.get(name, 0),
            refit=True,
        )
        gs.fit(X_train, y_train)
        elapsed = time.time() - t_start

        best_est = gs.best_estimator_

        t0 = time.time()
        best_est.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred_gs = best_est.predict(X_test)
        predict_time = time.time() - t0

        # SVM always uses decision_function (cuML SVC raises on predict_proba
        # unless probability=True was set at fit time)
        y_score_gs = (
            best_est.predict_proba(X_test)[:, 1]
            if name != "SVM" and hasattr(best_est, "predict_proba")
            else best_est.decision_function(X_test)
        )

        log.info("  Best params  : %s", gs.best_params_)
        log.info("  CV ROC-AUC   : %.4f", gs.best_score_)
        log.info("  Test Accuracy: %.4f", accuracy_score(y_test, y_pred_gs))
        log.info("  Test ROC-AUC : %.4f", roc_auc_score(y_test, y_score_gs))
        log.info("  Recall (yes) : %.4f", recall_score(y_test, y_pred_gs))
        log.info("  GridSearch   : %.2fs", elapsed)
        log.info("  Train time   : %.4fs", train_time)
        log.info("  Predict time : %.4fs", predict_time)

        results.append({
            "Model":            name,
            "Best Params":      gs.best_params_,
            "CV ROC-AUC":       round(gs.best_score_, 4),
            "Test Accuracy":    round(accuracy_score(y_test, y_pred_gs), 4),
            "Test ROC-AUC":     round(roc_auc_score(y_test, y_score_gs), 4),
            "Recall (yes)":     round(recall_score(y_test, y_pred_gs), 4),
            "F1 (yes)":         round(f1_score(y_test, y_pred_gs), 4),
            "GridSearch (s)":   round(elapsed, 2),
            "Train Time (s)":   round(train_time, 4),
            "Predict Time (s)": round(predict_time, 4),
            "best_estimator":   best_est,
            "y_pred":           y_pred_gs,
            "y_score":          y_score_gs,
        })
    return results


log.info("")
log.info("--- Running BASELINE (no resampling) ---")
gs_base_results = run_gridsearch(base_pipes, base_param_grids, "Base")

log.info("")
log.info("--- Running SMOTE ---")
gs_smote_results = run_gridsearch(smote_pipes, smote_param_grids, "SMOTE")

# ===========================================================================
# 9. Summary tables
# ===========================================================================
section("Summary - Baseline vs SMOTE")

exclude_keys = ("Best Params", "best_estimator", "y_pred", "y_score")
base_df  = pd.DataFrame([{k: v for k, v in r.items() if k not in exclude_keys}
                          for r in gs_base_results])
smote_df = pd.DataFrame([{k: v for k, v in r.items() if k not in exclude_keys}
                          for r in gs_smote_results])

log.info("Baseline (no resampling):\n%s", base_df.to_string(index=False))
log.info("")
log.info("SMOTE:\n%s", smote_df.to_string(index=False))

# Recall delta
log.info("")
log.info("Minority-class recall improvement (SMOTE - Baseline):")
for _, b_row in base_df.iterrows():
    s_row  = smote_df[smote_df["Model"] == b_row["Model"]].iloc[0]
    delta  = s_row["Recall (yes)"] - b_row["Recall (yes)"]
    prefix = "+" if delta >= 0 else ""
    log.info("  %-22s: %.4f -> %.4f  (%s%.4f)",
             b_row["Model"], b_row["Recall (yes)"], s_row["Recall (yes)"], prefix, delta)

# ===========================================================================
# 10. Comparison plots - Baseline vs SMOTE
# ===========================================================================
section("Saving comparison plots")

colors      = ["steelblue", "darkorange", "forestgreen", "crimson"]
model_names = base_df["Model"].tolist()
x           = np.arange(len(model_names))
width       = 0.35

# (a) Grouped bar chart: CV ROC-AUC, Test ROC-AUC, Recall (yes)
metrics_to_plot = [
    ("CV ROC-AUC",   "CV ROC-AUC (GridSearchCV)", 0.5, 1.0),
    ("Test ROC-AUC", "Test ROC-AUC",               0.5, 1.0),
    ("Recall (yes)", "Recall - Minority Class",     0.0, 1.0),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SMOTE vs Baseline - GridSearchCV Best Estimators", fontsize=13)
for ax, (metric, title, ymin, ymax) in zip(axes, metrics_to_plot):
    base_vals  = base_df[metric].values
    smote_vals = smote_df[metric].values
    b1 = ax.bar(x - width / 2, base_vals,  width, label="Baseline",
                color="lightsteelblue", edgecolor="steelblue")
    b2 = ax.bar(x + width / 2, smote_vals, width, label="SMOTE",
                color="salmon",         edgecolor="crimson")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(ymin, ymax); ax.set_title(title); ax.legend()
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
save_fig("comparison_baseline_vs_smote")

# (b) ROC curves: SMOTE (solid) vs Baseline (dashed)
fig, ax = plt.subplots(figsize=(8, 6))
for r, color in zip(gs_smote_results, colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_score"])
    ax.plot(fpr, tpr, color=color, linestyle="-",
            label=f"{r['Model']} SMOTE (AUC={r['Test ROC-AUC']:.3f})")
for r, color in zip(gs_base_results, colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_score"])
    ax.plot(fpr, tpr, color=color, linestyle="--", alpha=0.55,
            label=f"{r['Model']} Base  (AUC={r['Test ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - SMOTE (solid) vs Baseline (dashed)")
ax.legend(fontsize=8)
plt.tight_layout()
save_fig("comparison_roc_curves")

# (c) Train vs Predict time (SMOTE best estimators)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_times = smote_df["Train Time (s)"]
pred_times  = smote_df["Predict Time (s)"]
axes[0].barh(smote_df["Model"], train_times, color=colors)
axes[0].set_xlabel("Time (seconds)"); axes[0].set_title("SMOTE - Training Time (Best Estimator)")
for i, v in enumerate(train_times):
    axes[0].text(v + 0.0005, i, f"{v:.4f}s", va="center")
axes[1].barh(smote_df["Model"], pred_times, color=colors)
axes[1].set_xlabel("Time (seconds)"); axes[1].set_title("SMOTE - Prediction Time (Test Set)")
for i, v in enumerate(pred_times):
    axes[1].text(v + 0.0001, i, f"{v:.4f}s", va="center")
plt.tight_layout()
save_fig("comparison_train_predict_time")

# ===========================================================================
# 11. Logistic Regression coefficient plot (SMOTE best estimator)
# ===========================================================================
section("Logistic Regression - Coefficient Plot (SMOTE)")

best_lr_smote = gs_smote_results[0]["best_estimator"]
ohe_features  = (
    best_lr_smote.named_steps["preprocessor"]
                 .named_transformers_["onehotencoder"]
                 .get_feature_names_out(cat_cols)
)
feature_names = num_cols + list(ohe_features)
coefs = best_lr_smote.named_steps["model"].coef_[0]
coef_df = (
    pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    .sort_values("Coefficient", ascending=False)
)
bar_colors = ["steelblue" if c > 0 else "crimson" for c in coef_df["Coefficient"]]
fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.28)))
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Logistic Regression Coefficients - SMOTE-trained\n"
             "(positive = increases P(yes), negative = decreases P(yes))")
ax.invert_yaxis()
plt.tight_layout()
save_fig("lr_smote_coefficients")

# ===========================================================================
# 12. Persist trained models (baseline + SMOTE)
# ===========================================================================
section("Persisting trained models")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "02-smote")
os.makedirs(MODELS_DIR, exist_ok=True)

_filename_map = {
    "Logistic Regression": "logistic_regression",
    "KNN":                 "knn",
    "Decision Tree":       "decision_tree",
    "SVM":                 "svm",
}

metadata = {
    "dataset":         os.path.basename(DATA_PATH),
    "script":          os.path.relpath(
        __file__,
        start=os.path.join(os.path.dirname(__file__), ".."),
    ),
    "timestamp":       datetime.now().isoformat(timespec="seconds"),
    "svc_backend":     _SVC_BACKEND,
    "sklearn_version": sklearn.__version__,
    "python_version":  platform.python_version(),
    "notes": (
        "Each model has two variants: baseline (no resampling) and SMOTE. "
        "SVM was trained with the {backend} backend. "
        "Reloading svm variants requires {pkg} in the active environment; "
        "LR/KNN/DT reload on any scikit-learn install."
    ).format(
        backend=_SVC_BACKEND,
        pkg="RAPIDS cuML" if _SVC_BACKEND == "cuml" else "scikit-learn",
    ),
    "models": {},
}

for label, results in [("baseline", gs_base_results), ("smote", gs_smote_results)]:
    for r in results:
        stem  = _filename_map[r["Model"]]
        fname = f"{label}_{stem}.joblib"
        fpath = os.path.join(MODELS_DIR, fname)
        joblib.dump(r["best_estimator"], fpath)
        size_kb = os.path.getsize(fpath) / 1024
        log.info("  Saved %-12s %-20s -> %s  (%.1f KB)", label, r["Model"], fname, size_kb)

        key = f"{label}/{r['Model']}"
        metadata["models"][key] = {
            "file":          fname,
            "best_params":   r["Best Params"],
            "cv_roc_auc":    r["CV ROC-AUC"],
            "test_accuracy": r["Test Accuracy"],
            "test_roc_auc":  r["Test ROC-AUC"],
            "recall_yes":    r["Recall (yes)"],
            "f1_yes":        r["F1 (yes)"],
        }

metadata_path = os.path.join(MODELS_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, default=str)
log.info("  Metadata    -> %s", os.path.abspath(metadata_path))
log.info("All models persisted to: %s", os.path.abspath(MODELS_DIR))

# ===========================================================================
# 13. Speed and recall summary
# ===========================================================================
section("Speed and Recall Summary")

best_recall_model = smote_df.loc[smote_df["Recall (yes)"].idxmax(), "Model"]
best_auc_model    = smote_df.loc[smote_df["Test ROC-AUC"].idxmax(), "Model"]
fastest_train     = smote_df.loc[smote_df["Train Time (s)"].idxmin(), "Model"]
slowest_train     = smote_df.loc[smote_df["Train Time (s)"].idxmax(), "Model"]
fastest_pred      = smote_df.loc[smote_df["Predict Time (s)"].idxmin(), "Model"]
slowest_pred      = smote_df.loc[smote_df["Predict Time (s)"].idxmax(), "Model"]

log.info("Best minority-class recall (SMOTE) : %s", best_recall_model)
log.info("Best Test ROC-AUC        (SMOTE) : %s", best_auc_model)
log.info("Fastest to train                  : %s", fastest_train)
log.info("Slowest to train                  : %s", slowest_train)
log.info("Fastest to predict                : %s", fastest_pred)
log.info("Slowest to predict                : %s", slowest_pred)
log.info("")
log.info("Minority-class recall improvement (SMOTE - Baseline):")
for _, b_row in base_df.iterrows():
    s_row  = smote_df[smote_df["Model"] == b_row["Model"]].iloc[0]
    delta  = s_row["Recall (yes)"] - b_row["Recall (yes)"]
    prefix = "+" if delta >= 0 else ""
    log.info("  %-22s: %.4f -> %.4f  (%s%.4f)",
             b_row["Model"], b_row["Recall (yes)"], s_row["Recall (yes)"], prefix, delta)

log.info("")
log.info("Done. All outputs written to: %s", os.path.abspath(OUTPUT_DIR))

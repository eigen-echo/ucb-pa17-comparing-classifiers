"""
Training script for bank-additional-full.csv dataset.
Mirrors notebook 02-model-training-additional-full.ipynb.

Usage:
    python src/train_additional_full.py

Outputs:
    - Plots saved to outputs/02/
    - Log file at outputs/02/run.log
"""

import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Use non-interactive backend — no display required (safe for DGX Spark / headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup — writes to both console and a log file
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "02")
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
from sklearn.svm import LinearSVC  # noqa: F401 — kept for parity with notebook

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

# Shared preprocessor — sparse_output=False required for cuML SVC (needs dense input)
preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
)

# ===========================================================================
# 2. Dummy classifier — baseline
# ===========================================================================
section("Dummy Classifier (baseline)")

dummy_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
])
dummy_pipe.fit(X_train, y_train)
y_pred_dummy = dummy_pipe.predict(X_test)
y_prob_dummy = dummy_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred_dummy))
log.info("ROC-AUC  : %.4f", roc_auc_score(y_test, y_prob_dummy))

# ===========================================================================
# 3. Logistic Regression
# ===========================================================================
section("Logistic Regression")

lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train, y_train)
y_pred = lr_pipe.predict(X_test)
y_prob = lr_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred))
log.info("ROC-AUC  : %.4f", roc_auc_score(y_test, y_prob))
log.info("\n%s", classification_report(y_test, y_pred, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f"LR (AUC = {roc_auc_score(y_test, y_prob):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - Logistic Regression"); axes[1].legend()
plt.tight_layout()
save_fig("lr_confusion_roc")

log.info("Logistic Regression — 5-Fold Stratified CV")
cv_accuracy = cross_val_score(lr_pipe, X, y, cv=skf, scoring="accuracy")
cv_roc_auc  = cross_val_score(lr_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_accuracy.mean(), cv_accuracy.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_roc_auc.mean(), cv_roc_auc.std())

# ===========================================================================
# 4. K-Nearest Neighbors
# ===========================================================================
section("K-Nearest Neighbors")

knn_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier()),
])
knn_pipe.fit(X_train, y_train)
y_pred_knn = knn_pipe.predict(X_test)
y_prob_knn = knn_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred_knn))
log.info("ROC-AUC  : %.4f", roc_auc_score(y_test, y_prob_knn))
log.info("\n%s", classification_report(y_test, y_pred_knn, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - KNN")
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
axes[1].plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_score(y_test, y_prob_knn):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - KNN"); axes[1].legend()
plt.tight_layout()
save_fig("knn_confusion_roc")

log.info("KNN — 5-Fold Stratified CV")
cv_accuracy_knn = cross_val_score(knn_pipe, X, y, cv=skf, scoring="accuracy")
cv_roc_auc_knn  = cross_val_score(knn_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_accuracy_knn.mean(), cv_accuracy_knn.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_roc_auc_knn.mean(), cv_roc_auc_knn.std())

# ===========================================================================
# 5. Decision Tree
# ===========================================================================
section("Decision Tree")

dt_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42)),
])
dt_pipe.fit(X_train, y_train)
y_pred_dt = dt_pipe.predict(X_test)
y_prob_dt = dt_pipe.predict_proba(X_test)[:, 1]

log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred_dt))
log.info("ROC-AUC  : %.4f", roc_auc_score(y_test, y_prob_dt))
log.info("\n%s", classification_report(y_test, y_pred_dt, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - Decision Tree")
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
axes[1].plot(fpr_dt, tpr_dt, label=f"DT (AUC = {roc_auc_score(y_test, y_prob_dt):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - Decision Tree"); axes[1].legend()
plt.tight_layout()
save_fig("dt_confusion_roc")

log.info("Decision Tree — 5-Fold Stratified CV")
cv_accuracy_dt = cross_val_score(dt_pipe, X, y, cv=skf, scoring="accuracy")
cv_roc_auc_dt  = cross_val_score(dt_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_accuracy_dt.mean(), cv_accuracy_dt.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_roc_auc_dt.mean(), cv_roc_auc_dt.std())

# ===========================================================================
# 6. Support Vector Machine
# ===========================================================================
section("Support Vector Machine")
log.info("SVC backend: %s", _SVC_BACKEND)

svm_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVC(
        C=1.0,
        kernel="rbf",
        random_state=42,
        # no cap on GPU; restore original safeguard on CPU (uncapped sklearn SVC on ~40K rows never converges)
        max_iter=(-1 if _SVC_BACKEND == "cuml" else 2000),
    )),
])
svm_pipe.fit(X_train, y_train)
y_pred_svm = svm_pipe.predict(X_test)
y_score_svm = svm_pipe.decision_function(X_test)

log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred_svm))
log.info("ROC-AUC  : %.4f", roc_auc_score(y_test, y_score_svm))
log.info("\n%s", classification_report(y_test, y_pred_svm, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, display_labels=["no", "yes"], ax=axes[0])
axes[0].set_title("Confusion Matrix - SVM")
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
axes[1].plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_score(y_test, y_score_svm):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve - SVM"); axes[1].legend()
plt.tight_layout()
save_fig("svm_confusion_roc")

log.info("SVM — 5-Fold Stratified CV")
cv_accuracy_svm = cross_val_score(svm_pipe, X, y, cv=skf, scoring="accuracy")
cv_roc_auc_svm  = cross_val_score(svm_pipe, X, y, cv=skf, scoring="roc_auc")
log.info("  Accuracy : %.4f  (+/- %.4f)", cv_accuracy_svm.mean(), cv_accuracy_svm.std())
log.info("  ROC-AUC  : %.4f  (+/- %.4f)", cv_roc_auc_svm.mean(), cv_roc_auc_svm.std())

# ===========================================================================
# 7. GridSearchCV — all models
# ===========================================================================
section("GridSearchCV — Hyperparameter Tuning (all models)")

param_grids = {
    "Logistic Regression": {
        "pipe": lr_pipe,
        "params": {
            "model__C":      [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },
    "KNN": {
        "pipe": knn_pipe,
        "params": {
            "model__n_neighbors": [3, 5, 11, 21, 29],
            "model__weights":     ["uniform", "distance"],
            "model__metric":      ["euclidean", "manhattan"],
        },
    },
    "Decision Tree": {
        "pipe": dt_pipe,
        "params": {
            "model__max_depth":         [3, 5, 10, None],
            "model__min_samples_split": [2, 10, 20],
            "model__criterion":         ["gini", "entropy"],
        },
    },
    "SVM": {
        "pipe": svm_pipe,
        "params": {
            "model__C":      [0.01, 0.1, 1, 10, 100],
            "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "model__gamma":  ["scale", "auto"],
        },
    },
}

# GPU estimators don't parallelize well under joblib — one worker per GPU
per_model_n_jobs  = {"SVM": 1 if _SVC_BACKEND == "cuml" else -1}
# verbose=3 on GPU path: shows fold number (1/5, 2/5 ...) + params + time per fit
# verbose=2 shows params+time but omits the fold number
per_model_verbose = {"SVM": 3 if _SVC_BACKEND == "cuml" else 0}

gs_results = []

for name, config in param_grids.items():
    n_combinations = 1
    for v in config["params"].values():
        n_combinations *= len(v)
    total_fits = n_combinations * skf.n_splits
    log.info("Tuning %s  (%d combinations × %d folds = %d fits) ...",
             name, n_combinations, skf.n_splits, total_fits)
    t_start = time.time()

    gs = GridSearchCV(
        config["pipe"],
        config["params"],
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

    # SVM always uses decision_function: cuML SVC exposes predict_proba as a
    # method (so hasattr returns True) but raises at call time unless the model
    # was fitted with probability=True.  decision_function is always available.
    y_score_gs = (
        best_est.predict_proba(X_test)[:, 1]
        if name != "SVM" and hasattr(best_est, "predict_proba")
        else best_est.decision_function(X_test)
    )

    log.info("  Best params  : %s", gs.best_params_)
    log.info("  CV ROC-AUC   : %.4f", gs.best_score_)
    log.info("  Test Accuracy: %.4f", accuracy_score(y_test, y_pred_gs))
    log.info("  Test ROC-AUC : %.4f", roc_auc_score(y_test, y_score_gs))
    log.info("  GridSearch   : %.2fs", elapsed)
    log.info("  Train time   : %.4fs", train_time)
    log.info("  Predict time : %.4fs", predict_time)

    gs_results.append({
        "Model":            name,
        "Best Params":      gs.best_params_,
        "CV ROC-AUC":       round(gs.best_score_, 4),
        "Test Accuracy":    round(accuracy_score(y_test, y_pred_gs), 4),
        "Test ROC-AUC":     round(roc_auc_score(y_test, y_score_gs), 4),
        "GridSearch (s)":   round(elapsed, 2),
        "Train Time (s)":   round(train_time, 4),
        "Predict Time (s)": round(predict_time, 4),
        "best_estimator":   best_est,
        "y_pred":           y_pred_gs,
        "y_score":          y_score_gs,
    })

# ===========================================================================
# 8. Summary table
# ===========================================================================
section("Summary")

exclude_keys = ("Best Params", "best_estimator", "y_pred", "y_score")
summary_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in exclude_keys}
    for r in gs_results
])
log.info("\n%s", summary_df.to_string(index=False))

# ===========================================================================
# 9. Comparison plots
# ===========================================================================
section("Saving comparison plots")

colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
models      = summary_df["Model"]
test_acc    = summary_df["Test Accuracy"]
test_auc    = summary_df["Test ROC-AUC"]
gs_times    = summary_df["GridSearch (s)"]
train_times = summary_df["Train Time (s)"]
pred_times  = summary_df["Predict Time (s)"]

# Performance + GridSearch time
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].barh(models, test_acc, color=colors)
axes[0].set_xlim(0.4, 1.0); axes[0].set_xlabel("Test Accuracy")
axes[0].set_title("Test Accuracy by Model")
for i, v in enumerate(test_acc):
    axes[0].text(v + 0.001, i, f"{v:.4f}", va="center")
axes[1].barh(models, test_auc, color=colors)
axes[1].set_xlim(0.5, 1.0); axes[1].set_xlabel("Test ROC-AUC")
axes[1].set_title("Test ROC-AUC by Model")
for i, v in enumerate(test_auc):
    axes[1].text(v + 0.001, i, f"{v:.4f}", va="center")
axes[2].barh(models, gs_times, color=colors)
axes[2].set_xlabel("Time (seconds)"); axes[2].set_title("GridSearchCV Wall-Clock Time")
for i, v in enumerate(gs_times):
    axes[2].text(v + 0.5, i, f"{v:.1f}s", va="center")
plt.tight_layout()
save_fig("comparison_performance_gstime")

# Training vs Prediction time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(models, train_times, color=colors)
axes[0].set_xlabel("Time (seconds)"); axes[0].set_title("Training Time - Best Estimator")
for i, v in enumerate(train_times):
    axes[0].text(v + 0.0005, i, f"{v:.4f}s", va="center")
axes[1].barh(models, pred_times, color=colors)
axes[1].set_xlabel("Time (seconds)"); axes[1].set_title("Prediction Time - Best Estimator")
for i, v in enumerate(pred_times):
    axes[1].text(v + 0.0001, i, f"{v:.4f}s", va="center")
plt.tight_layout()
save_fig("comparison_train_predict_time")

# ROC curves overlaid
fig, ax = plt.subplots(figsize=(8, 6))
for r, color in zip(gs_results, colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_score"])
    ax.plot(fpr, tpr, label=f"{r['Model']} (AUC = {r['Test ROC-AUC']:.3f})", color=color)
ax.plot([0, 1], [0, 1], "k--", label="Baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - GridSearchCV Best Estimators"); ax.legend()
plt.tight_layout()
save_fig("comparison_roc_curves")

# ===========================================================================
# 10. Logistic Regression coefficient plot
# ===========================================================================
section("Logistic Regression — Coefficient Plot")

best_lr = gs_results[0]["best_estimator"]
ohe_features = (
    best_lr.named_steps["preprocessor"]
            .named_transformers_["onehotencoder"]
            .get_feature_names_out(cat_cols)
)
feature_names = num_cols + list(ohe_features)
coefs = best_lr.named_steps["model"].coef_[0]
coef_df = (
    pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    .sort_values("Coefficient", ascending=False)
)
bar_colors = ["steelblue" if c > 0 else "crimson" for c in coef_df["Coefficient"]]
fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.28)))
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Logistic Regression Coefficients\n(positive = increases P(yes), negative = decreases P(yes))")
ax.invert_yaxis()
plt.tight_layout()
save_fig("lr_coefficients")

# Speed summary
section("Speed Summary")
fastest_train = summary_df.loc[summary_df["Train Time (s)"].idxmin(), "Model"]
slowest_train = summary_df.loc[summary_df["Train Time (s)"].idxmax(), "Model"]
fastest_pred  = summary_df.loc[summary_df["Predict Time (s)"].idxmin(), "Model"]
slowest_pred  = summary_df.loc[summary_df["Predict Time (s)"].idxmax(), "Model"]
log.info("Fastest to train   : %s", fastest_train)
log.info("Slowest to train   : %s", slowest_train)
log.info("Fastest to predict : %s", fastest_pred)
log.info("Slowest to predict : %s", slowest_pred)
log.info("\n%s", summary_df[["Model", "Train Time (s)", "Predict Time (s)"]].to_string(index=False))

log.info("")
log.info("Done. All outputs written to: %s", os.path.abspath(OUTPUT_DIR))

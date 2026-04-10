# DGX Spark - GPU vs CPU SVM Findings

UC Berkeley Professional Certificate in ML/AI · Practical Application 17  
Out-of-scope learning exercise · branch `feature/setup-for-dgx-spark`

This document records the results of running `src/train_additional_full.py` on two hardware configurations: a Windows CPU laptop and an NVIDIA DGX Spark (Grace Blackwell GB10). The goal was to determine whether GPU-accelerated SVM via RAPIDS cuML, run without an iteration cap, changes the relative ranking of classifiers - and specifically whether a fully converged SVM can displace Logistic Regression as the recommended model.

---
## What was changed

- **SVC backend:** `sklearn.svm.SVC` replaced with `cuml.svm.SVC` (NVIDIA RAPIDS) via a dual-path import - the notebooks and script automatically fall back to sklearn on CPU machines.
- **Iteration cap removed:** `max_iter=2000` is lifted on the GPU path (`max_iter=-1`); the original cap is preserved on the CPU path to avoid the 6+ hour hang.
- **Expanded parameter grid:** `rbf` kernel restored (previously too slow on CPU) and `C` extended to `[0.01, 0.1, 1, 10, 100]` - 200 total fits vs. 60 before.
- **Standalone training script:** `src/train_additional_full.py` added as an alternative to papermill for headless execution on the Spark, with timestamped logging and PNG plot outputs.

---
## Hardware

| Property | CPU Machine (Windows) | DGX Spark (GPU) |
|----------|-----------------------|-----------------|
| **OS** | Windows 10 (Build 26200) | Ubuntu (Linux, ARM64) |
| **CPU** | Intel Core i9-12900HK (14 cores, 20 threads) | NVIDIA Grace (ARM64 aarch64) |
| **RAM** | 63.7 GB | 128 GB unified memory (CPU + GPU shared) |
| **GPU** | None | NVIDIA Blackwell GB10 (48 SMs, 6,144 CUDA cores) |
| **SVC Backend** | `sklearn.svm.SVC` (`max_iter=2000`) | `cuml.svm.SVC` (uncapped, `max_iter=-1`) |
| **Dataset** | `bank-additional-full.csv` (~41K rows, 20 features) | same |

---

## The Convergence Problem - Before and After

The original motivation for this exercise was that `sklearn.svm.SVC` on the ~41K row dataset could not converge within the `max_iter=2000` cap. The 5-fold cross-validation results from the baseline (pre-tuning) SVM on each machine tell the story clearly:

| Metric | CPU (sklearn, capped) | GPU (cuML, uncapped) |
|--------|-----------------------|----------------------|
| CV Accuracy | 0.4720 ± 0.0578 | **0.9014 ± 0.0019** |
| CV ROC-AUC | 0.3475 ± 0.0155 | **0.6980 ± 0.0091** |

A CV ROC-AUC of 0.3475 on CPU is **worse than random** - the capped model is not just suboptimal, it is actively producing inverted predictions in most folds, a clear sign the SMO optimiser halted midway through without settling on a valid decision boundary. The GPU run eliminates this entirely: accuracy and ROC-AUC are stable across folds and consistent with a properly fitted model.

![CPU: SVM confusion matrix and ROC curve - capped, degenerate](../outputs/02-cpu/svm_confusion_roc.png)

*CPU SVM single-fit: Accuracy 0.8561, ROC-AUC 0.6276. Surface-level accuracy looks acceptable because the model predicts the majority class most of the time, but the ROC-AUC is weak and the 5-fold CV reveals the instability.*

![GPU: SVM confusion matrix and ROC curve - converged, stable](../outputs/02-gpu/svm_confusion_roc.png)

*GPU SVM single-fit (cuML, rbf, C=1.0): Accuracy 0.9001, ROC-AUC 0.6782. Recall on the "yes" class rises from 0.19 (CPU) to 0.26 (GPU). Still imperfect, but the model has actually learned.*

---

## GridSearchCV Results - Side by Side

Both runs executed the same 5-fold stratified `GridSearchCV` over 40 SVM parameter combinations (C × kernel × gamma = 5 × 4 × 2), plus the same grids for LR, KNN, and Decision Tree.

### CPU (sklearn, n_jobs=-1 across 20 logical cores)

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC | GS Time |
|-------|:----------:|:-------------:|:------------:|:-------:|
| Logistic Regression | 0.7949 | 0.9000 | 0.7806 | 10.17s |
| KNN | 0.7714 | 0.8972 | 0.7537 | 61.75s |
| Decision Tree | 0.7846 | 0.9009 | 0.7721 | 5.17s |
| SVM | 0.6983 | 0.8401 | 0.7303 | **327s** |

> CPU SVM best params: `C=0.1`, `kernel=rbf`, `gamma=scale`

### GPU (cuML, n_jobs=1 - serialized to avoid GPU contention)

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC | GS Time |
|-------|:----------:|:-------------:|:------------:|:-------:|
| Logistic Regression | 0.7949 | 0.9000 | 0.7806 | 1.63s |
| KNN | 0.7748 | 0.8976 | 0.7634 | 14.25s |
| Decision Tree | 0.7846 | 0.9009 | 0.7721 | 1.55s |
| SVM | 0.7142 | 0.8949 | 0.7014 | **1,934s** |

> GPU SVM best params: `C=0.01`, `kernel=linear`, `gamma=scale`

![CPU: Test Accuracy, Test ROC-AUC, and GridSearch time comparison](../outputs/02-cpu/comparison_performance_gstime.png)

*CPU run: SVM grid search dominates at 327s but the quality (CV ROC-AUC 0.6983) is still well below LR.*

![GPU: Test Accuracy, Test ROC-AUC, and GridSearch time comparison](../outputs/02-gpu/comparison_performance_gstime.png)

*GPU run: SVM grid search balloons to 1,934s (32 minutes) - more than 6× slower than CPU in wall-clock terms despite per-fit GPU acceleration.*

---

## Why the GPU SVM Grid Search Is Slower

This is the most counterintuitive result of the experiment. A GPU should be faster - and per individual fit it is: the baseline single SVC fit completes in ~5s on the DGX Spark vs ~19s on the CPU laptop (**~3.8× faster per fit**). Yet the 200-fit grid search takes 1,934s on GPU vs 327s on CPU.

The explanation is parallelism:

- **CPU with `n_jobs=-1`:** sklearn's `GridSearchCV` distributes fits across all 20 logical processors via joblib. With 20 workers, 200 fits run in roughly 10 rounds of 20 concurrent fits. If each fit averages ~19s, total wall time is ~190–330s - matching the observed 327s.
- **GPU with `n_jobs=1`:** joblib is serialized to avoid multiple Python processes each allocating a CUDA context on the single Blackwell GPU. All 200 fits run sequentially. Even if each fit takes only 9.67s on average, 200 × 9.67s = 1,934s.

The GPU's 3.8× per-fit speedup is completely swamped by giving up 20× CPU parallelism. The GPU wins only on individual fit latency; the CPU wins on throughput via parallel folds.

**The practical takeaway:** GPU-accelerated SVM becomes advantageous over CPU parallelism when the per-fit speedup significantly exceeds the core-count ratio. On this dataset (~30K training rows, 20 features), that threshold is not met. A dataset 10–20× larger would likely cross the inflection point where cuML's per-fit latency advantage outweighs the serialization cost.

---

## Training and Prediction Time - Best Estimators

After GridSearchCV, each best estimator is re-fitted once on the full training set. These numbers measure deployment-relevant performance - how long it takes to train from scratch and score the test set.

### CPU

![CPU: Training time and prediction time by model](../outputs/02-cpu/comparison_train_predict_time.png)

| Model | Train Time | Predict Time |
|-------|:----------:|:------------:|
| Logistic Regression | 0.341s | 0.046s |
| KNN | 0.131s | **4.421s** |
| Decision Tree | 0.237s | 0.037s |
| SVM | **8.246s** | 5.045s |

*CPU KNN and SVM are slow predictors - KNN's 4.4s and SVM's 5.0s on a ~10K test set would be problematic for real-time scoring.*

### GPU

![GPU: Training time and prediction time by model](../outputs/02-gpu/comparison_train_predict_time.png)

| Model | Train Time | Predict Time |
|-------|:----------:|:------------:|
| Logistic Regression | 0.105s | 0.011s |
| KNN | 0.036s | 0.937s |
| Decision Tree | 0.076s | 0.011s |
| SVM | **0.546s** | **0.088s** |

*GPU SVM inference drops from 5.0s (CPU) to 0.088s - a 57× improvement. This is where GPU acceleration is unambiguous: once a cuML SVM is fitted, scoring is near-instant.*

The DGX Spark's ARM CPU also outperforms the Windows laptop for the sklearn models even without the GPU involved: LR trains in 0.105s vs 0.341s, DT in 0.076s vs 0.237s. These gains are from the stronger hardware platform and faster Linux I/O, not from CUDA.

---

## ROC Curves - Converged vs Uncapped

<table>
<tr>
<td align="center" width="50%">
<img src="../outputs/02-cpu/comparison_roc_curves.png" width="100%" alt="CPU ROC curves"/>
<em>CPU — sklearn (capped)<br>LR 0.781 · DT 0.772 · KNN 0.754 · SVM 0.730</em>
</td>
<td align="center" width="50%">
<img src="../outputs/02-gpu/comparison_roc_curves.png" width="100%" alt="GPU ROC curves"/>
<em>GPU — cuML (converged)<br>LR 0.781 · DT 0.772 · KNN 0.763 · SVM 0.701</em>
</td>
</tr>
</table>

The ordering is unchanged across both runs. The converged GPU SVM (0.701) lands slightly below the CPU SVM (0.730) in test ROC-AUC despite a higher CV ROC-AUC (0.714 vs 0.698), with the two runs selecting different kernels — `linear` on GPU vs `rbf` on CPU — suggesting both are near-equivalent local optima on this dataset.

---

## Key Technical Observations


The experiment is complete. Both a CPU (Windows, sklearn) run and a GPU (DGX Spark, cuML) run have been executed and compared. Key results:

- **Convergence confirmed.** The uncapped GPU SVM converges cleanly (CV ROC-AUC 0.698); the CPU-capped equivalent was degenerate (CV ROC-AUC 0.348 — below random). The iteration cap was masking a valid model, not a fundamental SVM limitation.
- **GPU grid search is slower in wall-clock time.** Per-fit training is ~3.8× faster on GPU, but serializing all 200 fits (`n_jobs=1` to avoid VRAM contention) means total GridSearchCV takes 1,934s on GPU vs 327s on CPU with parallelism across 20 cores.
- **GPU wins decisively on inference.** SVM prediction time drops from 5.0s (CPU) to 0.088s (GPU) on the ~10K test set — a 57× speedup relevant for scoring large client lists.
- **The original recommendation stands.** Even with full convergence, SVM (best Test ROC-AUC 0.730) does not displace Logistic Regression (0.781). The data's decision boundary is well-approximated linearly, which LR finds more efficiently.
- **Setup friction is real.** RAPIDS requires `mamba`, `scikit-learn` must be pinned to `1.5.x` (sklearn 1.6 introduced `__sklearn_tags__()` which cuML has not yet implemented), and cuML's `SVC.predict_proba` raises unless `probability=True` is set at fit time.
- **The two runs found different optimal kernels.** CPU GridSearchCV settled on `rbf` (C=0.1); GPU settled on `linear` (C=0.01). Their test ROC-AUC scores (0.730 vs 0.701) and CV ROC-AUC scores (0.698 vs 0.714) are close enough to suggest both kernels are near-equivalent on this dataset, and the rbf/linear choice at this scale is not decisive.
- **SVM prediction becomes deployment-viable on GPU.** CPU SVM inference on ~10K rows takes 5.0s; GPU takes 0.088s. If the use case required scoring millions of clients daily, a cuML-backed SVM endpoint would be practical where sklearn would not.
- **The macro-economic features drive LR's advantage over SVM.** The `euribor3m` and `nr.employed` features are dominant LR predictors (visible in the coefficient plot) and are likely highly correlated with each other - a regime that suits linear models. SVM's kernel trick adds flexibility but the underlying signal is still essentially linear.
- **All non-SVM models run faster on the DGX Spark regardless of GPU.** The ARM-based Grace CPU, faster NVMe storage, and Linux environment reduce LR grid search from 10.2s to 1.6s and DT from 5.2s to 1.6s. These are platform gains that will be signficant for much larger datasets.

---

## Does This Change the Recommendation?

**No.** Even with a fully converged, properly tuned SVM, Logistic Regression remains the superior model on this dataset:

| Model | CV ROC-AUC (GPU run) | Test ROC-AUC (GPU run) | Gap vs LR |
|-------|:--------------------:|:----------------------:|:---------:|
| **Logistic Regression** | **0.7949** | **0.7806** | - |
| Decision Tree | 0.7846 | 0.7721 | −0.009 |
| KNN | 0.7748 | 0.7634 | −0.017 |
| SVM (converged) | 0.7142 | 0.7014 | **−0.079** |

The fully converged SVM trails LR by **~8 percentage points in ROC-AUC** - a meaningful gap, not a rounding difference. The original concern documented in the README was that an iteration-capped SVM gave an unfair comparison. That concern is now resolved: a properly tuned SVM on this dataset is simply weaker than LR. The linear kernel winning the GPU GridSearchCV is consistent with this - the data's decision boundary is well-approximated by a hyperplane, which LR finds more efficiently and with better probability calibration.

The findings from `docs/findings.md` - that Logistic Regression is the recommended model for the bank telemarketing dataset - stand unchanged. Hence, I am not recollecting what the business outcomes mean / imply here again.

---

## Cloud Equivalent and Rental Cost

The GB10 in the DGX Spark is a **personal AI desktop chip** - not a datacenter Blackwell (B100/B200). I was priveleged to have access to one but not everyone has one lying around. Its 48 SMs, 6,144 CUDA cores, and 128 GB unified CPU+GPU memory pool make it roughly comparable to a single H100 for cuML SVM workloads, though the H100's memory bandwidth (3.35 TB/s vs ~273 GB/s on the GB10 GPU portion) would make the 200-fit GridSearchCV meaningfully faster.

**Important caveat:** AWS and GCP sell H100/B200 in 8-GPU minimum instances, which is overkill for this workload. Smaller GPU cloud providers offering single-GPU rentals are the practical equivalent.

### Closest equivalents for a 2-hour run

| Option | Provider | GPU | VRAM | Price/hr | 2-hr cost | Notes |
|--------|----------|-----|:----:|:--------:|:---------:|-------|
| **Best value** | Lambda Labs / Vast.ai | H100 SXM | 80 GB | $1.49–2.49 | **~$3–5** | Single GPU; faster than GB10; RAPIDS-compatible |
| **Stay on AWS** | AWS `g5.xlarge` | A10G | 24 GB | ~$1.01 | **~$2** | Cheapest RAPIDS-compatible single-GPU on AWS; dataset fits in 24 GB |
| **Stay on AWS, faster** | AWS `g6.xlarge` | L4 | 24 GB | ~$0.80 | **~$1.60** | Similar to A10G; good cuML support |
| **Stay on GCP** | GCP `a2-highgpu-1g` spot | A100 40 GB | 40 GB | ~$1–2 | **~$2–4** | Spot pricing; A100 is faster than GB10 |
| **Full Blackwell (overkill)** | AWS `p6-b200.48xlarge` | 8× B200 | 1.4 TB | ~$114 | **~$228** | Datacenter-grade; 7 of 8 GPUs would idle for this workload |

### Practical recommendation

An **AWS `g5.xlarge`** ($1.01/hr) or a **single H100 on Lambda Labs / Vast.ai** ($1.49–2.49/hr) would reproduce this experiment for **under $5**. The H100's superior memory bandwidth means the SVM GridSearchCV that took 32 minutes on the GB10 would likely complete in 10–15 minutes, so 1 hour of rental time is sufficient rather than 2.

The unified memory architecture of the GB10 (128 GB shared CPU+GPU) is not a factor for this specific workload — the preprocessed training matrix for 41K rows fits comfortably within 24 GB of VRAM.

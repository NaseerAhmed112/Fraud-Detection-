# 🔍 Data Science Internship — Project 2
## Supervised Learning: Fraud Detection Pipeline | Industrial Training Kit

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-FF6B6B?style=for-the-badge&logo=python&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble-228B22?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-blue?style=for-the-badge)

**Powered by [DecodeLabs](https://www.decodelabs.tech) | Greater Lucknow, India**

</div>

---

## 📌 Short Description

> *"A model that classifies every transaction as legitimate achieves near-perfect accuracy while resulting in catastrophic financial loss."*

**Project 2** is the vital bridge between data wrangling and real-world machine learning deployment. The goal is to build a **leak-free fraud detection pipeline** on a highly imbalanced financial dataset (99.83% legitimate vs 0.17% fraudulent). This project demonstrates why **Accuracy is a misleading metric** for imbalanced problems, how **SMOTE** synthesizes minority class samples without cloning, and why **`imblearn.pipeline.Pipeline`** is the only production-safe way to apply resampling inside cross-validation without data leakage.

---

## 🎯 Project Goal

**Build and tune a classification model to identify fraudulent transactions** in a highly imbalanced dataset — evaluated strictly on Precision, Recall, F1-Score, and ROC-AUC. Accuracy is completely discarded.

---

## 🏗️ Pipeline Architecture

```
Raw Imbalanced Data  ──►  Stratified  ──►  imblearn Pipeline  ──►  GridSearchCV  ──►  Final
(99.83% / 0.17%)          Split             [Scaler → SMOTE        (Multi-fold         Evaluation
                          (80/20)            → Classifier]          Tuning)            (ROC-AUC)
```

---

## 📂 Repository Structure

```
data-science-internship-project-2/
│
├── 📓 Project2_Fraud_Detection_Pipeline.ipynb  ← Main notebook (run this)
├── 📊 creditcard.csv                            ← Kaggle Credit Card Fraud dataset
├── 📄 README.md                                 ← You are here
│
├── outputs/
│   ├── class_imbalance.png                      ← 99.83% vs 0.17% visualization
│   ├── roc_auc_curve.png                        ← ROC-AUC comparison (LR vs RF)
│   ├── confusion_matrix_lr.png                  ← Logistic Regression results
│   ├── confusion_matrix_rf.png                  ← Random Forest results
│   └── feature_importance.png                   ← Random Forest feature weights
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | Credit Card Fraud Detection |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Total Transactions** | 284,807 |
| **Legitimate Rate** | 99.83% (284,315 transactions) |
| **Fraudulent Rate** | 0.17% (492 transactions) |
| **Features** | 30 (V1–V28 PCA-transformed + Time + Amount) |
| **Target** | `Class` — 0: Legitimate, 1: Fraudulent |

> **The Core Problem:** A model predicting "Legitimate" for every single transaction achieves **99.83% Accuracy** — yet catches **zero fraud**. This is the Accuracy Illusion Trap.

---

## 🔬 Key Engineering Concepts

### ⚠️ Trap #1 — The Illusion of Accuracy

| Metric | What It Measures | Why It Fails Here |
|--------|-----------------|-------------------|
| **Accuracy** | Overall correct predictions | Rewards predicting majority class always |
| **Precision** `TP / (TP+FP)` | When we flag fraud, are we right? | Minimizes false declines & customer frustration |
| **Recall** `TP / (TP+FN)` | Did we catch ALL the fraud? | Missing fraud = direct financial loss |
| **ROC-AUC** | Separation between distributions | Overall model capability — target: **0.85+** |

```
❌ Never optimize on Accuracy for imbalanced problems.
✅ Always optimize on Recall, F1-Score, and ROC-AUC.
```

---

### ⚠️ Trap #2 — The Data Leakage Catastrophe

Applying SMOTE **before** the train/test split means synthetic samples derived from training data leak into the test set — the model is evaluated on data it has already "seen" indirectly.

```
❌ WRONG ORDER (causes leakage):
   Entire Dataset → SMOTE → Train/Test Split

✅ CORRECT ORDER (zero leakage):
   Train/Test Split → SMOTE applied ONLY on training fold
```

---

### 🔹 Rebalancing — Why SMOTE Wins

| Method | Nickname | Problem |
|--------|----------|---------|
| **Undersampling** | The Loss | Destroys valuable baseline majority data |
| **Oversampling** | The Echo | Mere duplication → severe model overfitting |
| **SMOTE** | The Synthesis ✅ | Interpolates new synthetic minority points |

**SMOTE Interpolation Formula:**
```
x_new = x_i + λ × (x_nn - x_i)       where λ ~ Uniform(0, 1)
```
A random interpolation weight populates sparse regions of the minority feature space, helping classifiers learn a robust decision boundary.

---

### 🔹 The API Imperative — imblearn vs sklearn Pipeline

| API | Result | Why |
|-----|--------|-----|
| `sklearn.pipeline.Pipeline` | ❌ Fails in production | Standard steps only modify X — resampling (which modifies y too) is ignored or crashes |
| `imblearn.pipeline.Pipeline` | ✅ Production standard | Natively supports `fit_resample` — modifies both X and y **strictly on the training fold** |

```python
# ✅ Correct — imblearn Pipeline (leak-free)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('scaler',     StandardScaler()),   # must live INSIDE pipeline
    ('smote',      SMOTE()),            # applied only on training fold
    ('classifier', LogisticRegression())
])

# Random Forest Pipeline (no scaler needed — tree models are scale-invariant)
rf_pipeline = Pipeline([
    ('smote',      SMOTE()),
    ('classifier', RandomForestClassifier())
])
```

---

### 🔹 Model Comparison

| Model | Scaling Sensitivity | Decision Boundary | Best For |
|-------|--------------------|--------------------|---------|
| **Logistic Regression** | ⚠️ Fatal without StandardScaler — regularization penalties distorted by large Amount variances | Linear | Interpretable baseline, coefficient transparency |
| **Random Forest** | ✅ Immune — splits based on ordinal feature partitions, invariant to scale | Highly complex, non-linear | High-dimensional fraud patterns, feature importance |

---

### 🔹 Hyperparameter Tuning — GridSearchCV

`GridSearchCV` safely applies SMOTE **inside every fold** for every parameter combination — ensuring **zero leakage during hyperparameter tuning**.

```python
from sklearn.model_selection import GridSearchCV

# Logistic Regression param grid
lr_param_grid = {
    'smote__k_neighbors' : [3, 5, 7],
    'classifier__C'      : [0.01, 0.1, 1.0],
}

# Random Forest param grid
rf_param_grid = {
    'smote__k_neighbors'          : [3, 5, 7],
    'classifier__max_depth'       : [10, 20, None],
    'classifier__n_estimators'    : [100, 200],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring  = 'recall',       # optimize for catching ALL fraud
    cv       = 5,              # 5-fold stratified CV
    n_jobs   = -1
)
```

---

### 🔹 The Golden Rule of Validation

```
✅ NEVER expose your validation fold to SMOTE or StandardScaler.
✅ Your blind test set must reflect the REAL-WORLD extreme imbalance.
✅ SMOTE chamber operates strictly on the 80% training fold only.
```

---

## ✅ The Zero-Leakage Protocol (Complete Checklist)

```
☑  Ditch Accuracy — optimize using Recall, F1, and ROC-AUC
☑  Use SMOTE to interpolate and generate, NEVER just duplicate
☑  NEVER apply SMOTE or Scalers before the Train/Test Split
☑  ALWAYS use imblearn.pipeline.Pipeline (not sklearn's)
☑  Tune preprocessing AND model hyperparameters inside GridSearchCV
☑  Evaluate final model on untouched, real-world imbalanced test data
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/NaseerAhmed112/data-science-internship-project-2.git
cd data-science-internship-project-2

# 2. Download dataset from Kaggle
# Place creditcard.csv in the project root directory
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter

# 4. Launch Jupyter
jupyter notebook Project2_Fraud_Detection_Pipeline.ipynb

# 5. Run All → Kernel → Restart & Run All
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `Pandas` | 2.0+ | Data loading and manipulation |
| `NumPy` | 1.24+ | Numerical operations |
| `Scikit-learn` | 1.3+ | Logistic Regression, Random Forest, GridSearchCV, metrics |
| `Imbalanced-learn` | 0.11+ | SMOTE, `imblearn.pipeline.Pipeline` |
| `Matplotlib` | 3.7+ | ROC curves, confusion matrices |
| `Seaborn` | 0.12+ | Class distribution, heatmaps |

---

## 📊 Results Summary

```
═══════════════════════════════════════════════════════════════
  PROJECT 2: FRAUD DETECTION PIPELINE — FINAL RESULTS
═══════════════════════════════════════════════════════════════
  Dataset          : Credit Card Fraud (284,807 transactions)
  Class Imbalance  : 99.83% Legitimate / 0.17% Fraudulent
  Resampling       : SMOTE (k_neighbors tuned via GridSearchCV)
  Pipeline         : imblearn.pipeline.Pipeline (zero leakage)
─────────────────────────────────────────────────────────────
  Logistic Regression  →  ROC-AUC: ~0.97  |  Recall: ~0.92
  Random Forest        →  ROC-AUC: ~0.99  |  Recall: ~0.85+
─────────────────────────────────────────────────────────────
  Primary Metric   : ROC-AUC (target 0.85+) ✅
  Accuracy         : DISCARDED ❌ (misleading on imbalanced data)
═══════════════════════════════════════════════════════════════
```

---

## 📚 Key Skills Demonstrated

```
✅ Imbalanced Dataset Handling (99.83% / 0.17% split)
✅ SMOTE Synthetic Minority Over-sampling (interpolation)
✅ imblearn Pipeline (leak-free resampling in CV folds)
✅ Logistic Regression with StandardScaler
✅ Random Forest Classifier (scale-invariant ensemble)
✅ GridSearchCV Hyperparameter Tuning (Recall-optimized)
✅ Stratified Train/Test Split (preserving class ratios)
✅ Precision, Recall, F1-Score, ROC-AUC Evaluation
✅ Confusion Matrix Analysis (False Negatives = financial loss)
✅ Zero-Leakage Protocol (SMOTE strictly post-split)
✅ Feature Importance Analysis (Random Forest)
```

---

## 🏢 About

| | |
|--|--|
| **Program** | Data Science Industrial Training Kit |
| **Project** | Project 2 — Vital Bridge |
| **Track** | Supervised Learning & Fraud Detection |
| **Organization** | DecodeLabs |
| **Batch** | 2026 |
| **Location** | Greater Lucknow, India |
| **Contact** | decodelabs.tech@gmail.com |
| **Website** | [www.decodelabs.tech](https://www.decodelabs.tech) |

---

## 📜 License

This project is part of the DecodeLabs Industrial Training Program and is submitted for educational and certification purposes.

---

<div align="center">

**Made with ❤️ | DecodeLabs Batch 2026**

*"Your journey to becoming a professional Data Scientist accelerates right here, right now, with the very first classification algorithm you train today."*

</div>

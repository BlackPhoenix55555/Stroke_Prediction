# Stroke Prediction – Deployed Predictive Model

An end‑to‑end machine learning pipeline that predicts the risk of stroke from patient health data, deployed as a live REST API. The project is deliberately framed as a **business‑critical classification task** — directly analogous to **lead scoring**, **enrollment prediction**, and **retention modeling** in a product or marketing context.

**Key achievement:** Built a custom synthetic data augmentation strategy to overcome severe class imbalance (<5% positive cases) and tuned the model to prioritize **recall (catching positives)** over accuracy, exactly as you would when identifying at‑risk learners or churning customers.

---

## Business Use Case

- **Problem:** In any subscription or enrollment business, only a tiny fraction of users churn, convert, or fall into a high‑value segment. Standard accuracy becomes misleading.
- **Solution:** Treat the minority class (stroke = churn / lead conversion) as the **target to capture**. Maximize recall while keeping the model honest and deployable.
- **Direct analogs:** Lead scoring, retention prediction, customer lifetime value tiering.

---

## Dataset

- **Source:** Public stroke prediction dataset (Kaggle / Healthcare data)
- **Features:** Gender, Age, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Average Glucose Level, BMI, Smoking Status
- **Target:** Stroke (1 = stroke, 0 = healthy)
- **Imbalance:** ~5% positive class – severe skew

---

## Methodology

### Custom Data Augmentation (No SMOTE, No External Resamplers)

Class imbalance was addressed with a hand‑crafted augmentation strategy specifically designed to respect the **independence assumption of Naive Bayes**:

- **Inverse Transform Sampling (ECDF)** – Empirical CDF built per continuous feature from the minority class; uniform random numbers are inverted through linear interpolation, producing realistic samples within the original data range.
- **Convolution‑Based Smoothing** – Tiny Gaussian jitter added to the ECDF samples (`σ = 0.01 × feature_std`) to avoid exact duplicates and simulate natural variance.
- **Box‑Muller Gaussian Generation** – For features with near‑normal distributions (e.g., BMI), synthetic values are generated parametrically using the Box‑Muller transform.
- **Discrete Probability Sampling** – Categorical features are sampled from the observed probability mass function of the minority class.
- **Compositional Sampling** – All features are drawn independently, then concatenated. This compositional design perfectly matches the conditional independence assumption of Naive Bayes.

This approach expands the minority class to a **fully balanced training set** without any third‑party imbalance‑learn libraries.

### Dimensionality Reduction

- **PCA** applied after StandardScaler, retaining **95% of the variance** to reduce noise and improve generalization.

### Classifier

- **Gaussian Naive Bayes** – chosen for its alignment with the compositional augmentation and its simplicity, interpretability, and speed.

### Decision Threshold Tuning

- Instead of the default 0.5 threshold, the final model uses a **lower threshold (0.3)** to aggressively capture minority class instances.
- This **sacrifices overall accuracy** but **boosts recall to ~83%**, directly mimicking a business environment where missing a potential stroke (or a churning learner) is far more costly than a false alarm.

---

## Key Results (Honest, Unbiased)

| Metric | Value |
|--------|-------|
| **Recall (stroke=1)** | **0.83** |
| **ROC‑AUC** | **0.77** |
| Precision (stroke=1) | 0.10 |
| Accuracy | 0.66 |

> ℹ️ The low accuracy is **intentional** – it reflects the model’s focus on capturing almost all positive cases in a highly imbalanced setting. This is the correct trade‑off for churn/lead‑scoring problems.

---

## API Usage

The model is deployed as a **Flask REST API**. Send a JSON payload to the `/predict` endpoint.

### Endpoint

import numpy as np
import csv
import math
from collections import defaultdict

# ---------------- LOAD DATA ----------------
data = []
with open("stroke_data_clean.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        data.append([float(x.replace('"','')) for x in row])

data = np.array(data)

X = data[:, :-1]
y = data[:, -1].astype(int)

n_samples, n_features = X.shape

# ---------------- PRIOR ----------------
classes, counts = np.unique(y, return_counts=True)
P_class = {c: counts[i]/n_samples for i, c in enumerate(classes)}

# ---------------- FEATURE TYPES ----------------
categorical_idx = [0,2,3,4,5,6,9]   # same as your features
continuous_idx = [7,8]

# ---------------- CONDITIONAL PROB ----------------
P_conditional = defaultdict(lambda: defaultdict(dict))

for cls in classes:
    X_c = X[y == cls]

    for j in categorical_idx:
        values, val_counts = np.unique(X_c[:, j], return_counts=True)

        total = len(X_c)

        for v, count in zip(values, val_counts):
            P_conditional[j][v][cls] = count / total

# ---------------- GAUSSIAN PARAMS ----------------
Gaussian_params = {}

for j in continuous_idx:
    Gaussian_params[j] = {}
    for cls in classes:
        X_c = X[y == cls]
        Gaussian_params[j][cls] = {
            "mean": np.mean(X_c[:, j]),
            "std": np.std(X_c[:, j]) + 1e-6
        }

# ---------------- GAUSSIAN FUNCTION ----------------
def gaussian_pdf(x, mean, std):
    return (1 / (math.sqrt(2*math.pi)*std)) * math.exp(-((x-mean)**2)/(2*std**2))

# ---------------- PREDICTION ----------------
y_pred = []
y_prob = []

for i in range(n_samples):
    x = X[i]

    scores = {}

    for cls in classes:
        prob = P_class[cls]

        # categorical
        for j in categorical_idx:
            val = x[j]
            prob *= P_conditional[j].get(val, {}).get(cls, 1e-6)

        # continuous
        for j in continuous_idx:
            mean = Gaussian_params[j][cls]["mean"]
            std = Gaussian_params[j][cls]["std"]
            prob *= gaussian_pdf(x[j], mean, std)

        scores[cls] = prob

    total_score = sum(scores.values())
    prob_1 = scores[1] / (total_score + 1e-9)

    y_prob.append(prob_1)
    y_pred.append(1 if scores[1] > scores[0] else 0)

y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ---------------- METRICS ----------------
TP = np.sum((y_pred == 1) & (y == 1))
FP = np.sum((y_pred == 1) & (y == 0))
TN = np.sum((y_pred == 0) & (y == 0))
FN = np.sum((y_pred == 0) & (y == 1))

accuracy = (TP + TN) / len(y)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y, y_prob)
print("ROC-AUC:", auc)

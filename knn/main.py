import numpy as np
import math

data = []
with open("stroke_bigdata.csv") as f:
    for line in f:
        nums = [float(x.replace('"', '')) for x in line.strip().split(',')]
        data.append(nums)
data = np.array(data)

X = data[:, :-1]
y = data[:, -1].astype(int)

def knn_predict(X_train, y_train, x_test, k):
    distances = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
    k_indices = distances.argsort()[:k]
    k_nearest_labels = y_train[k_indices]

    values, counts = np.unique(k_nearest_labels, return_counts=True)
    hard_label = values[np.argmax(counts)]

    prob = np.mean(k_nearest_labels)  # since labels are 0/1

    return hard_label, prob


k = 3
y_pred = []
y_prob = []

for i in range(len(X)):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)

    hard, prob = knn_predict(X_train, y_train, X[i], k)
    y_pred.append(hard)
    y_prob.append(prob)

y_pred = np.array(y_pred)
y_prob = np.array(y_prob)
y_true = y.copy()

TP = np.sum((y_pred == 1) & (y_true == 1))
FP = np.sum((y_pred == 1) & (y_true == 0))
TN = np.sum((y_pred == 0) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))

accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
specificity = TN / (TN + FP + 1e-9)
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

brier = np.mean((y_true - y_prob)**2)

y_prob_clip = np.clip(y_prob, 1e-9, 1-1e-9)
log_loss = -np.mean(y_true*np.log(y_prob_clip) + (1-y_true)*np.log(1-y_prob_clip))

print("\nConfusion Matrix:")
print(f"[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")

print(f"Accuracy     = {accuracy*100:.2f}%")
print(f"Precision    = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print(f"Specificity  = {specificity:.4f}")
print(f"F1 Score     = {f1:.4f}")
print(f"Brier Score  = {brier:.6f}")
print(f"Log Loss     = {log_loss:.6f}")
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y, y_prob)
print("ROC-AUC:", auc)


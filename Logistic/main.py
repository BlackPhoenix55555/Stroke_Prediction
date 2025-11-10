import numpy as np
import math
import csv

dataset = []
with open("stroke_data_clean.csv") as f:
    for line in f:
        nums = [float(x.replace('"', '')) for x in line.strip().split(',')]
        dataset.append(nums)
dataset = np.array(dataset)

X = dataset[:, :-1]
y = dataset[:, -1]

X = np.hstack([np.ones((X.shape[0], 1)), X])

weights = np.zeros(X.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

learning_rate = 0.000001
tolerance = 0.00001

new_weights = weights.copy()

iteration = 0

while True:
    y_pred = sigmoid(X @ weights)
    error = y_pred - y

    gradient = (X.T @ error) / len(X)

    new_weights = weights - learning_rate * gradient

    if np.max(np.abs(new_weights - weights)) < tolerance:
        print("Converged at iteration:", iteration)
        weights = new_weights
        break

    weights = new_weights
    iteration += 1

predictions = (sigmoid(X @ weights) >= 0.5).astype(int)

pred = predictions.astype(int)
true = y.astype(int)

TP = np.sum((pred == 1) & (true == 1))
FP = np.sum((pred == 1) & (true == 0))
TN = np.sum((pred == 0) & (true == 0))
FN = np.sum((pred == 0) & (true == 1))

accuracy = (TP + TN) / len(y)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
specificity = TN / (TN + FP + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

probs = sigmoid(X @ weights)

brier = np.mean((true - probs)**2)

probs_clip = np.clip(probs, 1e-9, 1 - 1e-9)
logloss = -np.mean(true * np.log(probs_clip) + (1 - true) * np.log(1 - probs_clip))

ranks = probs.argsort().argsort()
num_pos = np.sum(true == 1)
num_neg = np.sum(true == 0)
auc = (np.sum(ranks[true == 1]) - (num_pos*(num_pos - 1))/2) / (num_pos * num_neg + 1e-9)

print("\nConfusion Matrix:")
print(f"[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")
print(f"Accuracy     = {accuracy*100:.2f}%")
print(f"Precision    = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print(f"Specificity  = {specificity:.4f}")
print(f"F1 Score     = {f1:.4f}")
print(f"Brier Score  = {brier:.6f}")
print(f"Log Loss     = {logloss:.6f}")
print(f"ROC-AUC      = {auc:.4f}")

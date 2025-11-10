import numpy as np
import math

def entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y == 1)
    p0 = 1 - p1
    if p1 == 0 or p0 == 0:
        return 0
    return -p1 * math.log2(p1) - p0 * math.log2(p0)

def information_gain(X_col, y, split_value):
    left_mask = X_col <= split_value
    right_mask = X_col > split_value

    left_y = y[left_mask]
    right_y = y[right_mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    parent_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)

    left_weight = len(left_y) / len(y)
    right_weight = len(right_y) / len(y)

    return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_value = None

    n_samples, n_features = X.shape

    for feature in range(n_features):
        values = np.unique(X[:, feature])
        for val in values:
            gain = information_gain(X[:, feature], y, val)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = val

    return best_feature, best_value, best_gain


class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label   # probability of class=1


def build_tree(X, y):
    if len(np.unique(y)) == 1:
        return Node(label=float(y[0]))  # pure 0 or 1

    feature, value, gain = find_best_split(X, y)

    if gain == 0:
        return Node(label=np.mean(y))  # probability leaf

    left_mask = X[:, feature] <= value
    right_mask = X[:, feature] > value

    left_child = build_tree(X[left_mask], y[left_mask])
    right_child = build_tree(X[right_mask], y[right_mask])

    return Node(feature, value, left_child, right_child)


def predict_one(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature] <= node.value:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


# ------------------- DATA LOADING -------------------
data = []
with open("stroke_reduced.csv") as f:
    for line in f:
        nums = [float(x.replace('"', '')) for x in line.strip().split(',')]
        data.append(nums)

data = np.array(data)
X = data[:, :-1]
y = data[:, -1].astype(int)

indices = np.arange(len(X))
np.random.shuffle(indices)
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

tree = build_tree(X_train, y_train)

# Get probabilities then convert to class prediction
y_prob_test = predict(tree, X_test)
preds_test = (y_prob_test >= 0.5).astype(int)

# ------------------- EVALUATION -------------------
TP = FP = TN = FN = 0
y_true = y_test
y_prob = y_prob_test

for i in range(len(y_test)):
    if preds_test[i] == 1 and y_test[i] == 1:
        TP += 1
    elif preds_test[i] == 1 and y_test[i] == 0:
        FP += 1
    elif preds_test[i] == 0 and y_test[i] == 0:
        TN += 1
    elif preds_test[i] == 0 and y_test[i] == 1:
        FN += 1

accuracy = (TP + TN) / len(y_test)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
specificity = TN / (TN + FP + 1e-9)

brier = np.mean((y_true - y_prob) ** 2)

y_prob_clipped = np.clip(y_prob, 1e-9, 1 - 1e-9)
log_loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))

ranks = y_prob.argsort().argsort()
num_pos = np.sum(y_true == 1)
num_neg = np.sum(y_true == 0)
roc_auc = (np.sum(ranks[y_true == 1]) - (num_pos*(num_pos-1))/2) / (num_pos * num_neg + 1e-9)

print("\nConfusion Matrix:")
print(f"[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")
print(f"Accuracy     = {accuracy*100:.2f}%")
print(f"Precision    = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print(f"Specificity  = {specificity:.4f}")
print(f"F1 Score     = {f1:.4f}")
print("Brier Score  =", brier)
print("Log Loss     =", log_loss)
print("ROC-AUC Score =", roc_auc)


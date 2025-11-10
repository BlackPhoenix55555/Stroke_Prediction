import numpy as np
from collections import Counter
import random
# -------------------- GINI FUNCTION --------------------
def gini(y):
    counts = Counter(y)
    return 1 - sum((c/len(y))**2 for c in counts.values())

# -------------------- BEST SPLIT (RANDOM FEATURE + RANDOM THRESHOLD) --------------------
def best_split(X, y):
    n_features = X.shape[1]
    features = random.sample(range(n_features), max(1, int(np.sqrt(n_features))))

    best_feature, best_threshold, best_gini = None, None, 1

    for feature in features:
        thresholds = np.unique(X[:, feature])

        if len(thresholds) > 10:
            thresholds = np.random.choice(thresholds, 10, replace=False)

        for threshold in thresholds:
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]
            if len(left)==0 or len(right)==0:
                continue

            g = (len(left)/len(y))*gini(left) + (len(right)/len(y))*gini(right)

            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

# -------------------- DECISION TREE --------------------
class DecisionTree:
    def __init__(self, depth=3):
        self.depth = depth

    def fit(self, X, y):
        if self.depth == 0 or len(set(y)) == 1:
            self.label = Counter(y).most_common(1)[0][0]
            return

        feat, thr = best_split(X, y)
        if feat is None:
            self.label = Counter(y).most_common(1)[0][0]
            return

        self.feature = feat
        self.threshold = thr

        left_mask = X[:, feat] <= thr
        self.left = DecisionTree(self.depth - 1)
        self.right = DecisionTree(self.depth - 1)

        self.left.fit(X[left_mask], y[left_mask])
        self.right.fit(X[~left_mask], y[~left_mask])

    def predict(self, x):
        if hasattr(self, 'label'):
            return self.label
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)

# -------------------- RANDOM FOREST --------------------
class RandomForest:
    def __init__(self, n_trees=9, depth=4):
        self.n_trees = n_trees
        self.depth = depth

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree(self.depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([[tree.predict(x) for tree in self.trees] for x in X])
        return np.array([Counter(row).most_common(1)[0][0] for row in preds])

def forest_probabilities(model, X):
    probs = []
    for x in X:
        votes = [tree.predict(x) for tree in model.trees]
        probs.append(sum(votes) / len(votes))
    return np.array(probs)

# -------------------- EVALUATION METRICS --------------------
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TN, FP, FN, TP

def precision_score(TP, FP): return TP / (TP + FP + 1e-9)
def recall_score(TP, FN): return TP / (TP + FN + 1e-9)
def specificity_score(TN, FP): return TN / (TN + FP + 1e-9)
def f1_score(p, r): return 2*(p*r)/(p+r+1e-9)
def brier_score(y_true, y_prob): return np.mean((y_prob - y_true)**2)
def log_loss(y_true, y_prob):
    eps = 1e-9
    return -np.mean(y_true*np.log(y_prob+eps) + (1-y_true)*np.log(1-y_prob+eps))
def roc_auc_score(y_true, scores):
    order = np.argsort(scores)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted == 1)
    cum_neg = np.cumsum(y_sorted == 0)
    auc = np.sum(cum_pos[y_sorted == 0]) / (cum_pos[-1] * cum_neg[-1] + 1e-9)
    return auc

# -------------------- LOAD DATA --------------------
data = []
with open("stroke_data_clean.csv") as f:
    for line in f:
        nums = [float(x.replace('"', '')) for x in line.strip().split(',')]
        data.append(nums)

data = np.array(data)
X, y = data[:, :-1], data[:, -1].astype(int)
# ---------------- TRAIN ON FULL DATASET ----------------
model = RandomForest(n_trees=9, depth=4)
model.fit(X, y)

all_preds = model.predict(X)
all_prob = forest_probabilities(model, X)

# ---------------- EVALUATE ON FULL DATASET ----------------
TN, FP, FN, TP = confusion_matrix(y, all_preds)

precision = precision_score(TP, FP)
recall = recall_score(TP, FN)
specificity = specificity_score(TN, FP)
f1 = f1_score(precision, recall)
brier = brier_score(y, all_prob)
logloss = log_loss(y, all_prob)
auc = roc_auc_score(y, all_prob)

print("\nConfusion Matrix (Full Dataset Evaluation):")
print(f"[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")
print(f"Accuracy     = {(all_preds == y).mean()*100:.2f}%")
print(f"Precision    = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print(f"Specificity  = {specificity:.4f}")
print(f"F1 Score     = {f1:.4f}")
print(f"Brier Score  = {brier:.6f}")
print(f"Log Loss     = {logloss:.6f}")
print(f"ROC-AUC      = {auc:.4f}")


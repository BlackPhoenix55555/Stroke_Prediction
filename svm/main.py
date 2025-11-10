import numpy as np

# ---------------- Load Dataset ----------------
data = []
with open("stroke_data_clean.csv") as f:
    for line in f:
        nums = [float(x.replace('"', '')) for x in line.strip().split(',')]
        data.append(nums)

data = np.array(data)
X = data[:, :-1]
y = data[:, -1]

# Convert labels 0 -> -1, keep stroke=1 as +1
y = np.where(y == 0, -1, 1)

# Train-Test Split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ---------------- Linear SVM Class ----------------
class SVM:
    def __init__(self, lr=0.0001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.W) + self.b) < 1
                
                if condition:
                    self.W -= self.lr * (self.lambda_param * self.W - y[i] * X[i])
                    self.b += self.lr * y[i]
                else:
                    self.W -= self.lr * (self.lambda_param * self.W)

    def decision_score(self, X):
        return np.dot(X, self.W) + self.b

    def predict(self, X):
        return np.sign(self.decision_score(X))


# ---------------- Train Model ----------------
model = SVM(lr=0.0001, lambda_param=0.01, epochs=500)
model.fit(X_train, y_train)

# ---------------- Predictions ----------------
scores = model.decision_score(X_test)

# Convert scores → probability using Sigmoid
prob = 1 / (1 + np.exp(-scores))

# Convert prob → labels (0/1 instead of -1/+1)
pred = np.where(prob >= 0.5, 1, 0)
y_true = np.where(y_test == 1, 1, 0)

# ---------------- Evaluation Metrics ----------------
TP = np.sum((pred == 1) & (y_true == 1))
FP = np.sum((pred == 1) & (y_true == 0))
TN = np.sum((pred == 0) & (y_true == 0))
FN = np.sum((pred == 0) & (y_true == 1))

accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
specificity = TN / (TN + FP + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

# Brier Score
brier = np.mean((y_true - prob)**2)

# Log Loss
prob_clip = np.clip(prob, 1e-9, 1-1e-9)
log_loss = -np.mean(y_true*np.log(prob_clip) + (1-y_true)*np.log(1-prob_clip))

# ROC-AUC (Rank-based)
ranks = prob.argsort().argsort()
num_pos = np.sum(y_true == 1)
num_neg = np.sum(y_true == 0)
roc_auc = (np.sum(ranks[y_true == 1]) - (num_pos*(num_pos-1)/2)) / (num_pos*num_neg + 1e-9)

# ---------------- Print Results ----------------
print("\nConfusion Matrix:")
print(f"[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")

print(f"Accuracy     = {accuracy*100:.2f}%")
print(f"Precision    = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print(f"Specificity  = {specificity:.4f}")
print(f"F1 Score     = {f1:.4f}")
print(f"Brier Score  = {brier:.6f}")
print(f"Log Loss     = {log_loss:.6f}")
print(f"ROC-AUC      = {roc_auc:.6f}")


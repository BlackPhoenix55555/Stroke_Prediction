import csv
import math

# ---------- Step 1: Read Data ----------
def read_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if not row:
                continue
            pc1 = float(row[0])
            pc2 = float(row[1])
            label = float(row[2])
            X.append([pc1, pc2])
            y.append(label)
    return X, y


# ---------- Step 2: Compute Mean and Std for each feature per class ----------
def compute_gaussian_params(X, y):
    params = {}
    classes = set(y)
    for c in classes:
        # select rows belonging to class c
        X_c = [X[i] for i in range(len(X)) if y[i] == c]
        means = [sum(feature) / len(feature) for feature in zip(*X_c)]
        stds = []
        for j in range(len(X_c[0])):
            mean = means[j]
            variance = sum((x[j] - mean) ** 2 for x in X_c) / len(X_c)
            stds.append(math.sqrt(variance))
        params[c] = {'mean': means, 'std': stds}
    return params


# ---------- Step 3: Gaussian PDF ----------
def gaussian_pdf(x, mean, std):
    if std == 0:
        return 1e-6
    exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


# ---------- Step 4: Predict class ----------
def predict(X, priors, params):
    preds = []
    for x in X:
        posteriors = {}
        for c in priors.keys():
            prior = priors[c]
            likelihood = 1.0
            for j in range(len(x)):
                mean = params[c]['mean'][j]
                std = params[c]['std'][j]
                likelihood *= gaussian_pdf(x[j], mean, std)
            posteriors[c] = prior * likelihood
        preds.append(max(posteriors, key=posteriors.get))
    return preds


# ---------- Step 5: Accuracy ----------
def accuracy(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)


# ---------- Step 6: Main ----------
filename = "stroke_reduced.csv"  # your file name
X, y = read_data(filename)

# Compute priors
classes = set(y)
priors = {}
for c in classes:
    priors[c] = y.count(c) / len(y)

# Compute Gaussian parameters
params = compute_gaussian_params(X, y)

# Predict
y_pred = predict(X, priors, params)

# Evaluate
acc = accuracy(y, y_pred)

# Print results
print(f"Accuracy = {acc * 100:.2f}%")


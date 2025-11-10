import numpy as np

data = []
with open("stroke_data_clean.csv") as f:
    for line in f:
        nums = [float(x.replace('"', '')) for x in line.strip().split(',')]
        data.append(nums)

data = np.array(data)
X = data[:, :-1]
y = data[:, -1].astype(int)


def manual_smote(X, y, target_class=1):
    minority = X[y == target_class]
    majority = X[y != target_class]

    num_min = len(minority)
    num_maj = len(majority)
    num_new = num_maj - num_min

    new_samples = []
    for _ in range(num_new):
        i, j = np.random.randint(0, num_min, 2)
        alpha = np.random.rand()
        new_samples.append(minority[i] + alpha * (minority[j] - minority[i]))

    return np.array(new_samples), np.array([target_class]*num_new)


new_minority, new_labels = manual_smote(X, y, 1)


X_balanced = np.vstack([X, new_minority])
y_balanced = np.hstack([y, new_labels])


def add_noise(X, noise_strength=0.01):
    noise = np.random.normal(0, noise_strength, X.shape)
    return X + noise

X_noisy = add_noise(X_balanced)


multiplier =1
X_big = np.tile(X_noisy, (multiplier, 1))
y_big = np.tile(y_balanced, multiplier)

print("Original size:", X.shape)
print("Balanced size:", X_balanced.shape)
print("BigData size:", X_big.shape)
print("Stroke=1 count:", np.sum(y_big == 1))
print("Stroke=0 count:", np.sum(y_big == 0))
# -------- Combine X and y before saving --------
bigdata = np.hstack([X_big, y_big.reshape(-1, 1)])   # make last column = label

# -------- Save to CSV --------
np.savetxt("stroke_bigdata.csv", bigdata, delimiter=",", fmt="%.5f")

print("✅ Big Data has been saved as stroke_bigdata.csv")


import numpy as np
import csv
import math

P_class = {
    1: 0.40965,
    0: 0.59014
}

P_conditional = {
    'Gender': {
        0: {1: 0.04142, 0: 0.95857},
        1: {1: 0.04425, 0: 0.95574},
        2: {1: 0.5, 0: 1.0}
    },
    'hypertension': {
        0: {1: 0.03342, 0: 0.96657},
        1: {1: 0.13303, 0: 0.88470}
    },
    'heart_disease': {
        0: {1: 0.03621, 0: 0.96378},
        1: {1: 0.16460, 0: 0.83539}
    },
    'ever_married': {
        0: {1: 0.01348, 0: 0.98651},
        1: {1: 0.94194, 0: 0.05805}
    },
    'work_type': {
        0: {1: 0.04517, 0: 0.95482},
        1: {1: 0.06838, 0: 0.93161},
        2: {1: 0.04444, 0: 0.95555},
        3: {1: 0.00149, 0: 0.99850},
        4: {1: 0.02272, 0: 1.0}
    },
    'Residence_type': {
        0: {1: 0.04377, 0: 0.95622},
        1: {1: 0.04133, 0: 0.95866}
    },
    'smoking status': {
        0: {1: 0.06810, 0: 0.93189},
        1: {1: 0.045464, 0: 0.95464},
        2: {1: 0.04535, 0: 0.94708},
        3: {1: 0.01955, 0: 0.98044}
    }
}

Gaussian_params = {
    'avg_glucose_level': {
        0: {'mean': 104.0, 'std': 40.0},
        1: {'mean': 134.57, 'std': 45.0}
    },
    'bmi': {
        0: {'mean': 28.8, 'std': 6.0},
        1: {'mean': 30.47, 'std': 5.0}
    }
}

def gaussian_pdf(x, mean, std):
    if std == 0:
        return 1e-6
    exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

def manual_log_loss(y_true, y_prob):
    total_loss = 0
    N = len(y_true)

    for i in range(N):
        p = y_prob[i]
        y = y_true[i]
        if y == 1:
            loss = math.log(p)
        else:
            loss = math.log(1 - p)

        total_loss += loss

    log_loss = - total_loss / N
    return log_loss

def brier_score(y_true, y_prob):
    N = len(y_true)
    total_error = 0
    
    for i in range(N):
        total_error += (y_true[i] - y_prob[i]) ** 2
        
    return total_error / N

def roc_auc_score_manual(y_true, y_prob):
    data = list(zip(y_true, y_prob))
    data.sort(key=lambda x: x[1])  

    ranks = []
    for rank, (label, prob) in enumerate(data, start=1):
        ranks.append((label, rank))

    sum_pos_ranks = sum(rank for label, rank in ranks if label == 1)

    n_pos = y_true.count(1)
    n_neg = y_true.count(0)
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg) 
    return auc

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask])
            ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)
    return ece

def maximum_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    max_error = 0.0
    
    for i in range(n_bins):
        bin_mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask])
            error = np.abs(bin_accuracy - bin_confidence)
            if error > max_error:
                max_error = error
    return max_error

def mean_squared_error(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return np.mean((y_true - y_prob)**2)

def mean_absolute_error(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return np.mean(np.abs(y_true - y_prob))

def root_mean_squared_error(y_true, y_prob):
    return np.sqrt(mean_squared_error(y_true, y_prob))

def cohen_kappa_manual(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    po = (TP + TN) / total
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total ** 2)
    if (1 - pe) == 0:
        return 0  # avoid division by zero
    kappa = (po - pe) / (1 - pe)
    return kappa


# +
def log_likelihood(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    #y_prob = np.clip(y_prob, eps, 1 - eps)
    ll = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))    
    return ll


# -

def prediction_entropy(y_prob):
    y_prob = np.clip(np.array(y_prob))
    
    entropy = - (y_prob * np.log(y_prob) + (1 - y_prob) * np.log(1 - y_prob))
    
    return np.mean(entropy)



filename = "stroke_data_clean.csv"

TP = FP = TN = FN = 0
total = 0

y_prob=[]
y_true=[]

with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue

        true_class = int(row[10])
        y_true.append(true_class)

        x = {
            'Gender': int(row[0]),
            'hypertension': int(row[2]),
            'heart_disease': int(row[3]),
            'ever_married': int(row[4]),
            'work_type': int(row[5]),
            'Residence_type': int(row[6]),
            'avg_glucose_level': float(row[7]),
            'bmi': float(row[8]),
            'smoking status': float(row[9])
        }

        scores = {}
        for cls in [0, 1]:
            prob = P_class[cls]

            for feature in ['Gender', 'hypertension', 'heart_disease', 'ever_married', 
                            'work_type', 'Residence_type', 'smoking status']:
                value = x[feature]
                if value in P_conditional[feature]:
                    prob *= P_conditional[feature][value][cls]
                else:
                    prob *= 1e-6  

            for feature in ['avg_glucose_level', 'bmi']:
                val = x[feature]
                mean = Gaussian_params[feature][cls]['mean']
                std = Gaussian_params[feature][cls]['std']
                prob *= gaussian_pdf(val, mean, std)

            scores[cls] = prob

        score_sum = scores[0] + scores[1]
        prob_class_1 = scores[1] / score_sum
        y_prob.append(prob_class_1)

        predicted_class = max(scores, key=scores.get)

        if predicted_class == 1 and true_class == 1:
            TP += 1
        elif predicted_class == 1 and true_class == 0:
            FP += 1
        elif predicted_class == 0 and true_class == 0:
            TN += 1
        elif predicted_class == 0 and true_class == 1:
            FN += 1

        total += 1

accuracy = (TP + TN) / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"Confusion Matrix:\n[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")
print(f"Accuracy     = {accuracy * 100:.2f}%")
print(f"Precision    = {precision:.4f}")
print(f"Recall (TPR) = {recall:.4f}")
print("Brier Score =", brier_score(y_true, y_prob))
print("Log Loss =", manual_log_loss(y_true, y_prob))
print("ROC-AUC Score =", roc_auc_score_manual(y_true, y_prob))
print("Expected calibration error = ",expected_calibration_error(y_true,y_prob,n_bins=10))
print("Max calibration error = ",maximum_calibration_error(y_true, y_prob, n_bins=10))
print("MSE = ",mean_squared_error(y_true, y_prob))
print("MAE = ",mean_absolute_error(y_true, y_prob))
print("rmse = ",root_mean_squared_error(y_true, y_prob))
print("kappa stats =",cohen_kappa_manual(TP, TN, FP, FN))
print("log-likelihood =",log_likelihood(y_true, y_prob))
print("entropy prediction =",prediction_entropy(y_prob))



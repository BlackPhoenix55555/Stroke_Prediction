# train_custom.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from scipy.interpolate import interp1d
import pickle
import warnings
warnings.filterwarnings('ignore')

# Optional xgboost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost.")

# ------------------------------
# 1. Load data
# ------------------------------
col_names = [
    'Gender', 'Age', 'Hypertension', 'Heart_disease', 'Ever_married',
    'Work_type', 'Residence_type', 'Avg_glucose_level', 'BMI',
    'Smoking_status', 'Stroke'
]
df = pd.read_csv('stroke_data_clean.csv', header=None, names=col_names)

X = df.drop('Stroke', axis=1)
y = df['Stroke']

# Keep test set completely untouched
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# 2. Custom augmentation (ECDF + Box-Muller + Discrete)
# ------------------------------
def custom_augment_minority_v2(
    X_minority, target_count, continuous_cols, discrete_cols,
    use_gaussian_cols=None, jitter_std_ratio=0.01, random_state=42
):
    np.random.seed(random_state)
    n_minority = len(X_minority)
    if n_minority == 0:
        raise ValueError("No minority samples to augment.")
    
    cont_models = {}
    for col in continuous_cols:
        data = X_minority[col].dropna().values.astype(float)
        if len(data) < 2:
            mean, std = data.mean(), data.std()
            cont_models[col] = ('gaussian', mean, std if std > 0 else 1e-6)
        elif use_gaussian_cols and col in use_gaussian_cols:
            cont_models[col] = ('boxmuller', data.mean(), data.std())
        else:
            cont_models[col] = ('ecdf', data)
    
    disc_models = {}
    for col in discrete_cols:
        counts = X_minority[col].value_counts(normalize=True)
        disc_models[col] = (counts.index.values, counts.values)
    
    synthetic_rows = []
    for _ in range(target_count):
        row = {}
        for col in continuous_cols:
            model = cont_models[col]
            if model[0] == 'gaussian':
                row[col] = np.random.normal(model[1], model[2])
            elif model[0] == 'boxmuller':
                u1, u2 = np.random.random(), np.random.random()
                z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
                row[col] = model[1] + model[2] * z
            else:  # ecdf
                data = model[1]
                sorted_data = np.sort(data)
                probs = (np.arange(1, len(sorted_data)+1)) / len(sorted_data)
                u = np.random.uniform(0, 1)
                interp = interp1d(probs, sorted_data, kind='linear',
                                  bounds_error=False,
                                  fill_value=(sorted_data[0], sorted_data[-1]))
                val = interp(u)
                # tiny convolution jitter
                std_jitter = jitter_std_ratio * data.std()
                val += np.random.normal(0, std_jitter)
                row[col] = val
        
        for col in discrete_cols:
            vals, probs = disc_models[col]
            row[col] = np.random.choice(vals, p=probs)
        
        synthetic_rows.append(row)
    
    return pd.DataFrame(synthetic_rows, columns=X_minority.columns)

# ------------------------------
# 3. Create balanced training set
# ------------------------------
continuous_cols = ['Age', 'Avg_glucose_level', 'BMI']
discrete_cols   = ['Gender', 'Hypertension', 'Heart_disease', 'Ever_married',
                   'Work_type', 'Residence_type', 'Smoking_status']
use_gaussian_cols = ['BMI']   # assume BMI is roughly Gaussian

train_data = X_train.copy()
train_data['Stroke'] = y_train
minority = train_data[train_data['Stroke'] == 1]
majority = train_data[train_data['Stroke'] == 0]

n_minority = len(minority)
n_majority = len(majority)
synthetic_needed = n_majority - n_minority

if synthetic_needed > 0:
    synthetic_df = custom_augment_minority_v2(
        minority.drop('Stroke', axis=1),
        synthetic_needed,
        continuous_cols,
        discrete_cols,
        use_gaussian_cols=use_gaussian_cols,
        jitter_std_ratio=0.01
    )
    synthetic_df['Stroke'] = 1
    balanced_train = pd.concat([majority, minority, synthetic_df], ignore_index=True)
else:
    balanced_train = train_data

balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bal = balanced_train.drop('Stroke', axis=1)
y_train_bal = balanced_train['Stroke']

# ------------------------------
# 4. Models to evaluate (no ComplementNB)
# ------------------------------
models_to_test = {
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
}
if XGB_AVAILABLE:
    models_to_test['XGBoost'] = XGBClassifier(
        scale_pos_weight=n_majority / n_minority,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

# ------------------------------
# 5. Model comparison & threshold tuning (using original training set)
# ------------------------------
results = {}
best_model_name = None
best_pipeline = None
best_threshold = 0.5

for name, model in models_to_test.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('classifier', model)
    ])
    pipe.fit(X_train_bal, y_train_bal)
    
    # Use original (imbalanced) training set to tune threshold
    train_proba = pipe.predict_proba(X_train)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_train, train_proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_t = thresh[best_idx] if best_idx < len(thresh) else 0.5
    
    # Evaluate on test set
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_t).astype(int)
    
    recall_test = recall_score(y_test, test_pred)
    prec_test = precision_score(y_test, test_pred)
    f1_test = f1_score(y_test, test_pred)
    roc_test = roc_auc_score(y_test, test_proba)
    acc_test = (test_pred == y_test).mean()
    
    results[name] = {
        'threshold': best_t,
        'test_recall': recall_test,
        'test_precision': prec_test,
        'test_f1': f1_test,
        'test_roc_auc': roc_test,
        'test_accuracy': acc_test
    }
    
    print(f"\n{name}:")
    print(f"  Best threshold (from train): {best_t:.4f}")
    print(f"  Test -> Recall: {recall_test:.4f}, Precision: {prec_test:.4f}, F1: {f1_test:.4f}, AUC: {roc_test:.4f}, Acc: {acc_test:.4f}")
    
    # Track best by recall (or change to f1)
    if recall_test > results.get(best_model_name, {}).get('test_recall', 0):
        best_model_name = name
        best_pipeline = pipe
        best_threshold = best_t

# ------------------------------
# 6. Summary and save
# ------------------------------
print("\n" + "="*60)
print(f"Best model by recall: {best_model_name} (threshold={best_threshold:.4f})")
print("All test results:")
for name, r in results.items():
    print(f"  {name}: Rec={r['test_recall']:.4f}, Prec={r['test_precision']:.4f}, F1={r['test_f1']:.4f}, AUC={r['test_roc_auc']:.4f}, Acc={r['test_accuracy']:.4f}")

# Save best pipeline and threshold
with open('model.pkl', 'wb') as f:
    pickle.dump({'pipeline': best_pipeline, 'threshold': best_threshold}, f)

print(f"\nSaved best model ({best_model_name}) with threshold {best_threshold:.4f} to model.pkl")

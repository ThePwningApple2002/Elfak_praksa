import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix
)

import KitNET.KitNET as kit

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data...")
X_raw = pd.read_csv("/home/milan/Desktop/Elfak_praksa/Kitsune-py/archive/03-02-2018.csv", low_memory=False)

X_raw = X_raw.replace(["Infinity", "inf"], 0)
X_raw = X_raw.replace(np.nan, 0)
if "Timestamp" in X_raw.columns:
    X_raw = X_raw.drop(["Timestamp"], axis=1)

print("Applying filter: Sparse Fwd<=1")
X_raw = X_raw[(X_raw["TotLen Fwd Pkts"] <= 1)]

# ==========================================
# 2. Construct Train/Test Splits
# ==========================================
print("Restructuring dataset for training purity...")
df_benign = X_raw[X_raw["Label"] == 'Benign'].copy()
df_bot = X_raw[X_raw["Label"] == 'Bot'].copy()

total_benign = len(df_benign)
print(f"Total Benign available: {total_benign}")
print(f"Total Bot available: {len(df_bot)}")

Ratio = 0.85
train_size = int(total_benign * Ratio)
FMgrace = int(train_size * 0.25)
ADgrace = train_size - FMgrace

df_train = df_benign.iloc[:train_size].copy()

df_test_benign = df_benign.iloc[train_size:].copy()
df_test = pd.concat([df_test_benign, df_bot], axis=0)

df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

df_final = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

Labels = np.where(df_final["Label"] == 'Benign', 0, 1)
df_final = df_final.drop(["Label"], axis=1)

X_val = df_final.astype(np.float32).to_numpy()
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

print("\nDataset Structure:")
print(f"FMgrace (Feature Mapper): {FMgrace}")
print(f"ADgrace (Anomaly Detector): {ADgrace}")
print(f"Total execution/testing records: {len(df_test)}")

# ==========================================
# 3. Feature Selection (ExtraTrees)
# ==========================================
print("\nRunning ExtraTrees Feature Selection...")
clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, bootstrap=True, random_state=42)
clf = clf.fit(X_val, Labels)

model = SelectFromModel(clf, prefit=True)
X_selected = model.transform(X_val)

print(f"Reduced features from {X_val.shape[1]} to {X_selected.shape[1]}")

# ==========================================
# 4. Normalization
# ==========================================
norm1 = Normalizer(norm='l2')
X_norm = norm1.fit_transform(X_selected)

# ==========================================
# 5. KitNET Execution
# ==========================================
print("\nInitializing KitNET...")
K = kit.KitNET(X_norm.shape[1], FMgrace, ADgrace)
RMSEs = np.zeros(X_norm.shape[0])

print("Processing records...")
start = time.time()
for i in range(X_norm.shape[0]):
    if i % 10000 == 0 and i > 0:
        print(f"Processed {i} records...")
    
    RMSEs[i] = K.process(X_norm[i], X_norm)

stop = time.time()
print(f"KitNET Complete. Time elapsed: {stop - start:.2f} seconds")

exec_start = FMgrace + ADgrace
X_test_scores = RMSEs[exec_start:].reshape(-1, 1)
y_test_labels = Labels[exec_start:]

# ==========================================
# 6. Clustering & Evaluation
# ==========================================
print("\nClustering Anomaly Scores with K-Means...")
clf_kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
y_pred = clf_kmeans.fit_predict(X_test_scores)

center_0 = clf_kmeans.cluster_centers_[0][0]
center_1 = clf_kmeans.cluster_centers_[1][0]

if center_0 > center_1:
    print("Inverting K-Means labels (Cluster 0 had the higher RMSE).")
    y_pred = 1 - y_pred 
else:
    print("K-Means labels correctly aligned (Cluster 1 has the higher RMSE).")

# ==========================================
# 7. Metrics & Output
# ==========================================
acc = accuracy_score(y_test_labels, y_pred)
pre = precision_score(y_test_labels, y_pred, zero_division=0)
rec = recall_score(y_test_labels, y_pred, zero_division=0)
f1 = f1_score(y_test_labels, y_pred, zero_division=0)
roc = roc_auc_score(y_test_labels, y_pred)

print("\n--- Final Execution Phase Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC_AUC:   {roc:.4f}")

print("\nConfusion Matrix:")
print("[[TN  FP]")
print(" [FN  TP]]")
print(confusion_matrix(y_test_labels, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test_labels, y_pred).ravel()

fpr = fp / (fp + tn)

print(f"False Positive Rate: {fpr:.4f}")

FTP, TPR, threshold = roc_curve(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(FTP, TPR, color='darkorange', lw=2, label=f'ROC curve (area = {roc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Execution Phase ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
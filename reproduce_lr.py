"""
Replica de la metodología del avance del compañero (Regresión Logística,
split 80/20, pd.get_dummies drop_first=True, umbral 0.50) para poder
completar las secciones 3.2, 3.8 y 4 del informe con un modelo
consistente con lo ya escrito en las secciones 1-3.7.
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("telco_churn.csv")

# TotalCharges: 11 en blanco, todos con tenure=0 -> imputar con 0 (según el informe)
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn_bin"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID", "Churn"])

# Binarias con mapeo directo 0/1
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df[c] = df[c].map({"Yes": 1, "No": 0})

# One-Hot Encoding (drop_first=True) para el resto de categóricas
# (select_dtypes con "object" no detecta columnas dtype "str" en pandas >=3)
other_cat_cols = [c for c in df.columns if df[c].dtype in ("object", "str") and c != "Churn_bin"]
df = pd.get_dummies(df, columns=other_cat_cols, drop_first=True)

y = df["Churn_bin"]
X = df.drop(columns=["Churn_bin"])
print(f"Matriz de features: {X.shape[1]} columnas")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "n_test": len(y_test),
}
cm = confusion_matrix(y_test, y_pred)
print(json.dumps(metrics, indent=2))
print("Matriz de confusión:\n", cm)

# Figura 6: gráfico de la matriz de confusión (sección 3.7 del informe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
labels = ["No Churn", "Churn"]
ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusión (Test, n={len(y_test)})", fontsize=12)
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14)
plt.tight_layout()
plt.savefig("figs/fig6_confusion.png", dpi=130)
plt.close()
print("Figura 6 guardada: figs/fig6_confusion.png")

# Guardar todo lo necesario para el despliegue
joblib.dump(model, "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
joblib.dump(num_cols, "numeric_columns.pkl")
with open("repro_metrics.json", "w") as f:
    json.dump({"metrics": metrics, "confusion_matrix": cm.tolist()}, f, indent=2)
print("\nModelo y artefactos guardados.")

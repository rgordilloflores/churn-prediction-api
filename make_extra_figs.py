import warnings
warnings.filterwarnings("ignore")
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("telco_churn.csv")
os.makedirs("figs", exist_ok=True)
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float).fillna(0)
df["Churn_bin"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID", "Churn"])
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df[c] = df[c].map({"Yes": 1, "No": 0})
other_cat_cols = [c for c in df.columns if df[c].dtype in ("object", "str") and c != "Churn_bin"]
df = pd.get_dummies(df, columns=other_cat_cols, drop_first=True)

y = df["Churn_bin"]
X = df.drop(columns=["Churn_bin"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train_s, y_train)
y_proba = model.predict_proba(X_test_s)[:, 1]

# Figura 7: Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 4.8))
ax.plot(fpr, tpr, color="#2C5F8A", lw=2.2, label=f"Regresión Logística (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Clasificador aleatorio")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curva ROC - modelo final (test set)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figs/fig7_roc.png", dpi=130)
plt.close()

# Figura 8: Curva de aprendizaje (Recall vs tamaño de train, 5-fold CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    X_train_s, y_train, cv=cv, scoring="recall",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#55A868", label="Recall - Entrenamiento")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#C44E52", label="Recall - Validación (CV)")
ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color="#55A868")
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="#C44E52")
ax.set_xlabel("Tamaño del conjunto de entrenamiento")
ax.set_ylabel("Recall")
ax.set_title("Curva de aprendizaje - Regresión Logística")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figs/fig8_learning.png", dpi=130)
plt.close()

gap_inicial = train_scores.mean(axis=1)[0] - val_scores.mean(axis=1)[0]
gap_final = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
print(f"AUC test: {auc:.4f}")
print(f"Brecha train-val inicial: {gap_inicial:.4f} | final: {gap_final:.4f}")
print("Figuras guardadas: figs/fig7_roc.png, figs/fig8_learning.png")

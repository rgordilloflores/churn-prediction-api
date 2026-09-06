import warnings
warnings.filterwarnings("ignore")
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv("telco_churn.csv")
os.makedirs("figs", exist_ok=True)
df = df_raw.copy()
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float).fillna(0)
df["Churn_bin"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID", "Churn"])
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df[c] = df[c].map({"Yes": 1, "No": 0})
other_cat_cols = [c for c in df.columns if df[c].dtype in ("object", "str") and c != "Churn_bin"]
df_enc = pd.get_dummies(df, columns=other_cat_cols, drop_first=True)

y = df_enc["Churn_bin"]
X = df_enc.drop(columns=["Churn_bin"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
idx_test = X_test.index  # para recuperar MonthlyCharges original de estos clientes

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train_s, y_train)
y_proba = model.predict_proba(X_test_s)[:, 1]

# =====================================================================
# MEJORA 1: Optimización del umbral de decisión (Precision/Recall/F1 vs umbral)
# =====================================================================
thresholds = np.arange(0.05, 0.95, 0.02)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    pred_t = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, pred_t, zero_division=0))
    recalls.append(recall_score(y_test, pred_t, zero_division=0))
    f1s.append(f1_score(y_test, pred_t, zero_division=0))

best_f1_idx = int(np.argmax(f1s))
# Umbral que garantiza Recall >= 0.85 con la mejor Precision posible en ese rango
mask_recall85 = np.array(recalls) >= 0.85
idx_recall85 = np.where(mask_recall85)[0]
best_recall85_idx = idx_recall85[np.argmax(np.array(precisions)[idx_recall85])] if len(idx_recall85) else None

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(thresholds, precisions, label="Precisión", color="#065A82", lw=2)
ax.plot(thresholds, recalls, label="Recall", color="#C44E52", lw=2)
ax.plot(thresholds, f1s, label="F1-Score", color="#55A868", lw=2)
ax.axvline(0.50, color="gray", linestyle="--", alpha=0.6, label="Umbral actual (0.50)")
if best_recall85_idx is not None:
    ax.axvline(thresholds[best_recall85_idx], color="#F4A259", linestyle=":", lw=2,
               label=f"Umbral sugerido ({thresholds[best_recall85_idx]:.2f})")
ax.set_xlabel("Umbral de decisión")
ax.set_ylabel("Score")
ax.set_title("Optimización del umbral: Precisión vs. Recall vs. F1")
ax.legend(loc="center left", fontsize=9)
plt.tight_layout()
plt.savefig("figs/fig9_umbral.png", dpi=130)
plt.close()

threshold_summary = {
    "umbral_actual": 0.50,
    "umbral_mejor_f1": round(float(thresholds[best_f1_idx]), 2),
    "f1_en_mejor_f1": round(float(f1s[best_f1_idx]), 4),
    "recall_en_mejor_f1": round(float(recalls[best_f1_idx]), 4),
}
if best_recall85_idx is not None:
    threshold_summary.update({
        "umbral_sugerido_recall85": round(float(thresholds[best_recall85_idx]), 2),
        "recall_en_sugerido": round(float(recalls[best_recall85_idx]), 4),
        "precision_en_sugerido": round(float(precisions[best_recall85_idx]), 4),
        "f1_en_sugerido": round(float(f1s[best_recall85_idx]), 4),
    })
    # Impacto en conteo real de clientes
    pred_sugerido = (y_proba >= thresholds[best_recall85_idx]).astype(int)
    fn_sugerido = int(((pred_sugerido == 0) & (y_test == 1)).sum())
    fp_sugerido = int(((pred_sugerido == 1) & (y_test == 0)).sum())
    fn_actual = int(((y_proba >= 0.50).astype(int) == 0) & (y_test == 1)).sum() if False else int(((( y_proba >= 0.50).astype(int) == 0) & (y_test == 1)).sum())
    fp_actual = int((((y_proba >= 0.50).astype(int) == 1) & (y_test == 0)).sum())
    threshold_summary.update({
        "falsos_negativos_actual": fn_actual, "falsos_positivos_actual": fp_actual,
        "falsos_negativos_sugerido": fn_sugerido, "falsos_positivos_sugerido": fp_sugerido,
    })

print("=== OPTIMIZACIÓN DE UMBRAL ===")
print(json.dumps(threshold_summary, indent=2))

# =====================================================================
# MEJORA 2: Matriz de priorización Riesgo x Valor del cliente
# =====================================================================
monthly_value = df_raw.loc[idx_test, "MonthlyCharges"].values
median_value = np.median(monthly_value)

priority_df = pd.DataFrame({
    "churn_probability": y_proba,
    "monthly_value": monthly_value,
    "actual_churn": y_test.values,
})
priority_df["priority_score"] = priority_df["churn_probability"] * priority_df["monthly_value"]
priority_df["cuadrante"] = np.select(
    [
        (priority_df.churn_probability >= 0.5) & (priority_df.monthly_value >= median_value),
        (priority_df.churn_probability >= 0.5) & (priority_df.monthly_value < median_value),
        (priority_df.churn_probability < 0.5) & (priority_df.monthly_value >= median_value),
    ],
    ["Alto riesgo / Alto valor (URGENTE)", "Alto riesgo / Bajo valor (monitorear)", "Bajo riesgo / Alto valor (fidelizar)"],
    default="Bajo riesgo / Bajo valor (bajo esfuerzo)",
)

quadrant_counts = priority_df["cuadrante"].value_counts().to_dict()
print("\n=== MATRIZ RIESGO x VALOR (conteo por cuadrante, test set) ===")
print(json.dumps(quadrant_counts, indent=2, ensure_ascii=False))

colors = {
    "Alto riesgo / Alto valor (URGENTE)": "#C44E52",
    "Alto riesgo / Bajo valor (monitorear)": "#F4A259",
    "Bajo riesgo / Alto valor (fidelizar)": "#4C72B0",
    "Bajo riesgo / Bajo valor (bajo esfuerzo)": "#B0B0B0",
}
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for cuad, color in colors.items():
    sub = priority_df[priority_df.cuadrante == cuad]
    ax.scatter(sub.churn_probability, sub.monthly_value, s=18, alpha=0.6, color=color, label=f"{cuad} (n={len(sub)})")
ax.axvline(0.5, color="black", lw=1, alpha=0.4)
ax.axhline(median_value, color="black", lw=1, alpha=0.4)
ax.set_xlabel("Probabilidad de Churn")
ax.set_ylabel("Valor mensual del cliente (MonthlyCharges, USD)")
ax.set_title("Matriz de priorización: Riesgo de Churn x Valor del Cliente")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig("figs/fig10_riesgo_valor.png", dpi=130)
plt.close()

top10 = priority_df.sort_values("priority_score", ascending=False).head(10)
print("\n=== TOP 10 CLIENTES PRIORITARIOS (mayor ingreso mensual en riesgo) ===")
print(top10[["churn_probability", "monthly_value", "priority_score"]].round(3).to_string())

with open("threshold_and_priority_summary.json", "w") as f:
    json.dump({
        "threshold_summary": threshold_summary,
        "quadrant_counts": quadrant_counts,
        "ingreso_mensual_total_en_riesgo_alto_valor": round(
            float(priority_df[priority_df.cuadrante == "Alto riesgo / Alto valor (URGENTE)"]["monthly_value"].sum()), 2
        ),
    }, f, indent=2, ensure_ascii=False)
print("\nGuardado: threshold_and_priority_summary.json, figs/fig9_umbral.png, figs/fig10_riesgo_valor.png")

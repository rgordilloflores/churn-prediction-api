"""
Generación de las Figuras 1-5 del informe (Análisis Exploratorio de Datos).
Corresponde a la sección 2.2 del informe.

Ejecutar después de tener telco_churn.csv en la misma carpeta:
    python generar_figuras_eda.py
Genera: figs/fig1_balance.png ... figs/fig5_pago.png
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_csv("telco_churn.csv")
os.makedirs("figs", exist_ok=True)
df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn_bin"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------------------------------------------------------------------
# Figura 1: Distribución de la variable objetivo (Churn)
# ---------------------------------------------------------------------------
counts = df["Churn"].value_counts().reindex(["No", "Yes"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["No", "Yes"], counts.values, color=["#2c5f8a", "#c0392b"])
ymax = counts.max()
ax.set_ylim(0, ymax * 1.18)
for i, v in enumerate(counts.values):
    ax.text(i, v + ymax * 0.03, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=11)
ax.set_title("Distribución de la variable objetivo (Churn)", fontsize=13, pad=14)
ax.set_ylabel("N° de clientes")
plt.tight_layout()
plt.savefig("figs/fig1_balance.png", dpi=130)
plt.close()
print("Figura 1 generada: distribución de clases "
      f"(No={counts['No']}, {counts['No']/len(df)*100:.1f}% | "
      f"Yes={counts['Yes']}, {counts['Yes']/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Figura 2: Correlación de Pearson de variables numéricas con Churn
# ---------------------------------------------------------------------------
num_cols = ["MonthlyCharges", "SeniorCitizen", "TotalCharges_num", "tenure"]
corr = df[num_cols + ["Churn_bin"]].corr()["Churn_bin"].drop("Churn_bin")
corr = corr.reindex(["MonthlyCharges", "SeniorCitizen", "TotalCharges_num", "tenure"])
colors = ["#c0392b" if v > 0 else "#2c5f8a" for v in corr.values]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coeficiente de correlación de Pearson")
ax.set_title("Correlación de variables numéricas con Churn", fontsize=13)
plt.tight_layout()
plt.savefig("figs/fig2_correlacion.png", dpi=130)
plt.close()
print("Figura 2 generada: correlaciones ->", corr.round(3).to_dict())

# ---------------------------------------------------------------------------
# Figura 3: Distribución de antigüedad (tenure) según churn
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.arange(0, df["tenure"].max() + 3, 2)
ax.hist(df.loc[df.Churn == "No", "tenure"], bins=bins, alpha=0.6, label="No se fugó", color="#6a8caf")
ax.hist(df.loc[df.Churn == "Yes", "tenure"], bins=bins, alpha=0.6, label="Se fugó", color="#c0796f")
ax.set_xlabel("Meses como cliente")
ax.set_ylabel("N° de clientes")
ax.set_title("Antigüedad (tenure) según fuga", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig("figs/fig3_tenure.png", dpi=130)
plt.close()
print("Figura 3 generada: distribución de tenure por churn")

# ---------------------------------------------------------------------------
# Figura 4: Tasa de fuga por tipo de contrato
# ---------------------------------------------------------------------------
tasa_contrato = (df.groupby("Contract")["Churn_bin"].mean() * 100).reindex(
    ["Month-to-month", "One year", "Two year"]
)
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(tasa_contrato.index, tasa_contrato.values, color="#c0392b")
for i, v in enumerate(tasa_contrato.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11)
ax.set_ylabel("% de clientes que se fugaron")
ax.set_title("Tasa de fuga por tipo de contrato", fontsize=13)
plt.tight_layout()
plt.savefig("figs/fig4_contrato.png", dpi=130)
plt.close()
print("Figura 4 generada: tasa de fuga por contrato ->", tasa_contrato.round(1).to_dict())

# ---------------------------------------------------------------------------
# Figura 5: Tasa de fuga por método de pago
# ---------------------------------------------------------------------------
tasa_pago = (df.groupby("PaymentMethod")["Churn_bin"].mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(tasa_pago.index, tasa_pago.values, color="#8a5a2c")
for i, v in enumerate(tasa_pago.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("% de clientes que se fugaron")
ax.set_title("Tasa de fuga por método de pago", fontsize=13)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figs/fig5_pago.png", dpi=130)
plt.close()
print("Figura 5 generada: tasa de fuga por método de pago ->", tasa_pago.round(1).to_dict())

print("\nListo. Las 5 figuras de EDA se guardaron en figs/")

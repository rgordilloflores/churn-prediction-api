# Predicción de Churn — Código Completo del Proyecto

Maestría en Ciencia de Datos & IA — Curso de Machine Learning — UPN

Este paquete contiene TODO el código necesario para reproducir, desde cero,
cada número, tabla y gráfico citado en el informe — y para volver a desplegar
la API si hiciera falta.

## Requisitos previos

```bash
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn httpx joblib pydantic
```

Todos los scripts asumen que `telco_churn.csv` está en la misma carpeta desde
donde se ejecutan, y crean automáticamente una carpeta `figs/` si no existe.

## Cómo generar las 10 figuras del informe (en este orden)

| # | Script a ejecutar | Qué genera |
|---|---|---|
| 1 | `python generar_figuras_eda.py` | Figuras 1-5: distribución de churn, correlaciones numéricas, antigüedad vs. churn, tasa de fuga por contrato y por método de pago (sección 2.2) |
| 2 | `python reproduce_lr.py` | Entrena el modelo final (Regresión Logística), guarda `logreg_model.pkl` + Figura 6 (matriz de confusión, sección 3.7) |
| 3 | `python make_extra_figs.py` | Figura 7 (curva ROC) y Figura 8 (curva de aprendizaje), sección 3.7 |
| 4 | `python mejoras.py` | Figura 9 (optimización de umbral) y Figura 10 (matriz riesgo x valor), secciones 3.7 y 3.8 |

Ejecutarlos todos de corrido:
```bash
python generar_figuras_eda.py
python reproduce_lr.py
python make_extra_figs.py
python mejoras.py
```

Al terminar, la carpeta `figs/` debe tener exactamente 10 archivos
(`fig1_balance.png` ... `fig10_riesgo_valor.png`), listos para insertar en el
informe si se necesita regenerar alguno.

**Nota:** las Figuras 1-6 usan los mismos datos y lógica que originalmente
produjo el equipo en el análisis exploratorio; los números impresos en
consola (42.7%, 11.3%, 2.8% de fuga por contrato, etc.) deben coincidir
exactamente con los citados en el informe.

## Estructura de archivos

```
.
├── telco_churn.csv              # Dataset (IBM Telco Customer Churn)
├── generar_figuras_eda.py       # Figuras 1-5 (EDA)
├── reproduce_lr.py              # Entrenamiento del modelo final + Figura 6
├── make_extra_figs.py           # Figuras 7-8 (ROC, curva de aprendizaje)
├── mejoras.py                   # Figuras 9-10 (umbral, riesgo x valor)
├── figs/                        # Las 10 figuras generadas (se crea sola)
├── logreg_model.pkl, scaler.pkl, feature_columns.pkl, numeric_columns.pkl
├── repro_metrics.json, threshold_and_priority_summary.json, load_test_report.json
├── api/
│   ├── app.py                   # API REST de despliegue (FastAPI)
│   └── test_inference.py        # Pruebas de inferencia y carga (sección 4)
└── deploy_ready/                # Carpeta autocontenida lista para subir a GitHub + Render
    ├── app.py, requirements.txt, los 4 .pkl
    └── README_DESPLIEGUE.md     # Guía paso a paso de despliegue
```

## Cómo correr la API localmente

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8001
# en otra terminal:
python test_inference.py
```

## Cómo volver a desplegar la API a producción (Render, gratis)

Ver la guía completa paso a paso en `deploy_ready/README_DESPLIEGUE.md`.
Resumen rápido:

1. Subir el contenido de `deploy_ready/` a un repositorio de GitHub.
2. Crear cuenta gratuita en render.com (con "Sign up with GitHub").
3. New Web Service → conectar el repositorio.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Instance Type: Free → Deploy.
7. Probar `https://<tu-servicio>.onrender.com/health`.

**Despliegue real de este proyecto:**
- API en producción: https://churn-prediction-api-evip.onrender.com
- Documentación interactiva: https://churn-prediction-api-evip.onrender.com/docs
- Repositorio: https://github.com/rgordilloflores/churn-prediction-api

## Métricas finales (test set, n=1,409) — Regresión Logística, umbral 0.50

| Métrica | Valor |
|---|---|
| Recall | 0.783 |
| F1-Score | 0.614 |
| Precisión | 0.505 |
| Accuracy | 0.739 |
| AUC-ROC | 0.842 |

Con umbral ajustado a 0.41 (sección 3.7 del informe): Recall = 0.861.

Ver `Informe_Churn_ML_Completo.docx` para el análisis completo de cada figura
y `Guia_Entendimiento_Proyecto.docx` para una explicación en lenguaje simple
de todo el proyecto, pensada para la sustentación individual.

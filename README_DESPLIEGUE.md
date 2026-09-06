# Cómo desplegar esta API a una URL pública gratuita (Render.com)

## Paso 1: Subir esta carpeta a GitHub
1. Crea un repositorio nuevo en https://github.com/new (puede ser público o privado).
2. Sube TODOS los archivos de esta carpeta (app.py, requirements.txt, los 4 .pkl) a ese repositorio.
   - Esto además resuelve el Anexo 7 del informe (enlace al repositorio).

## Paso 2: Crear el servicio en Render
1. Entra a https://render.com y crea una cuenta gratuita (puedes usar tu cuenta de GitHub).
2. Click en "New +" → "Web Service".
3. Conecta tu repositorio de GitHub.
4. Configura así:
   - **Name**: churn-prediction-api (o el nombre que prefieran)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
5. Click "Create Web Service" y espera 2-3 minutos mientras se despliega.

## Paso 3: Probar que funciona
Cuando termine, Render te da una URL pública como:
`https://churn-prediction-api-xxxx.onrender.com`

Prueba en el navegador: `https://churn-prediction-api-xxxx.onrender.com/health`
Debe responder: `{"status":"ok","model_loaded":true,"modelo":"LogisticRegression"}`

## Paso 4: Avísame la URL
Compárteme la URL pública y actualizo la sección 3.8 del informe reemplazando
"validado localmente" por la URL real en producción.

**Nota:** el plan gratuito de Render "duerme" el servicio tras 15 min sin uso
y tarda ~30-50 segundos en despertar en la primera petición — esto es normal
y se puede mencionar en la sustentación como una limitación conocida del tier gratuito.

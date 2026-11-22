# Práctica 2: Máquinas de Soporte Vectorial (SVM) – Fine Tuning (4 Kernels × 4 Valores de $C$ = 16 Combinaciones) y MLflow

**Dataset:** FreshRetailNet-50K (600 registros, 10 productos, 60 días: `2024-03-28` a `2024-05-26`) | **Métricas:** RMSE, MAE, $R^2$ y Tiempo (ms)

## Estructura en 2 Cuadernos (`.ipynb` y `.html`)

1. **[`Experimentos_MLflow.ipynb`](Experimentos_MLflow.ipynb)** / **[`Experimentos_MLflow.html`](Experimentos_MLflow.html)** (Cuaderno 1 — Limpieza, 16 Combinaciones en MLflow y Registro de *Models*):
   - Realiza la limpieza de datos (verificación de nulos y duplicados), ingeniería de variables temporales desde `fecha_venta`, escalado (`StandardScaler`) y codificación (`OneHotEncoder`).
   - Evalúa **4 tipos de kernel** (`Lineal`, `Polinomico`, `RBF_Gaussiano`, `Sigmoide`) × **4 valores de regularización $C \in \{0.1, 1.0, 5.0, 10.0\}$**, registrando las **16 combinaciones exactas en MLflow** con sus 16 gráficas individuales de evaluación, gráficas comparativas y datasets.
   - Selecciona los mejores modelos y los registra oficialmente en la sección **Models (Model Registry)** de MLflow:
     - **`Mejor_Modelo_SVM_Global`** (`Exp07_SVM_Polinomico_C5` — RMSE: **574.37 u** | MAE: **303.42 u** | $R^2$: **0.9206**)
     - **`Mejor_SVM_Polinomico`** (`Exp07_SVM_Polinomico_C5`)
     - **`Mejor_SVM_RBF_Gaussiano`** (`Exp11_SVM_RBF_Gaussiano_C5`)
     - **`Mejor_SVM_Lineal`** (`Exp03_SVM_Lineal_C5`)
     - **`Mejor_SVM_Sigmoide`** (`Exp16_SVM_Sigmoide_C10`)

2. **[`Practica2.ipynb`](Practica2.ipynb)** / **[`Practica2.html`](Practica2.html)** (Cuaderno 2 — Predicción por Producto y Fecha vs. Valor Real, Nuevos Samples y Explicabilidad):
   - Carga los mejores modelos seleccionados en MLflow.
   - Implementa la función **`predecir_por_producto_y_fecha(product_id, fecha_venta)`**: al ingresar un producto y una fecha, predice la demanda y la muestra junto al **Valor Real (`Valor_Real (u)`)**, el **Error Absoluto** y el **Error Porcentual (`%`)**, con gráficas comparativas de barras y serie temporal.
   - Implementa **`predecir_nuevo_sample()`** para escenarios hipotéticos (evaluada en 4 casos de distintas categorías y análisis de sensibilidad al descuento).
   - Incluye explicabilidad con *Permutation Feature Importance*, **Conclusiones con resultados cuantitativos** y **Referencias en formato APA 7.ª edición**.

---

## Tabla de las 16 Combinaciones Registradas en MLflow (Ordenadas por Mejor RMSE)

| # | Experimento en MLflow | Kernel | $C$ | RMSE (u) | MAE (u) | $R^2$ | Tiempo (ms) | Registro en *Models* |
| :-: | :--- | :---: | :-: | :---: | :---: | :---: | :---: | :--- |
| 1 | **`Exp07_SVM_Polinomico_C5`** | **Polinómico ($d=2$)** | **5.0** | **574.37** | **303.42** | **0.9206** | 19.7 | **`Mejor_Modelo_SVM_Global` / `Mejor_SVM_Polinomico`** |
| 2 | `Exp08_SVM_Polinomico_C10` | Polinómico ($d=2$) | 10.0 | 599.74 | 308.43 | 0.9134 | 27.6 | — |
| 3 | **`Exp11_SVM_RBF_Gaussiano_C5`** | **RBF Gaussiano** | **5.0** | **609.45** | **313.09** | **0.9106** | 12.8 | **`Mejor_SVM_RBF_Gaussiano`** |
| 4 | `Exp12_SVM_RBF_Gaussiano_C10` | RBF Gaussiano | 10.0 | 635.56 | 320.56 | 0.9027 | 13.7 | — |
| 5 | `Exp06_SVM_Polinomico_C1` | Polinómico ($d=2$) | 1.0 | 902.29 | 403.71 | 0.8039 | 12.6 | — |
| 6 | **`Exp03_SVM_Lineal_C5`** | **Lineal** | **5.0** | **953.35** | **400.20** | **0.7811** | 38.3 | **`Mejor_SVM_Lineal`** |
| 7 | `Exp02_SVM_Lineal_C1` | Lineal | 1.0 | 961.96 | 400.86 | 0.7771 | 15.8 | — |
| 8 | `Exp04_SVM_Lineal_C10` | Lineal | 10.0 | 966.39 | 402.75 | 0.7751 | 54.0 | — |
| 9 | `Exp01_SVM_Lineal_C0.1` | Lineal | 0.1 | 1023.43 | 417.65 | 0.7478 | 12.7 | — |
| 10 | **`Exp16_SVM_Sigmoide_C10`** | **Sigmoide** | **10.0** | **1076.84** | **445.10** | **0.7207** | 11.4 | **`Mejor_SVM_Sigmoide`** |
| 11 | `Exp10_SVM_RBF_Gaussiano_C1` | RBF Gaussiano | 1.0 | 1103.75 | 472.77 | 0.7066 | 11.1 | — |
| 12 | `Exp15_SVM_Sigmoide_C5` | Sigmoide | 5.0 | 1265.99 | 510.19 | 0.6140 | 11.1 | — |
| 13 | `Exp05_SVM_Polinomico_C0.1` | Polinómico ($d=2$) | 0.1 | 1703.52 | 816.38 | 0.3011 | 11.7 | — |
| 14 | `Exp14_SVM_Sigmoide_C1` | Sigmoide | 1.0 | 1899.86 | 845.47 | 0.1308 | 10.9 | — |
| 15 | `Exp09_SVM_RBF_Gaussiano_C0.1` | RBF Gaussiano | 0.1 | 2020.40 | 902.76 | 0.0170 | 10.9 | — |
| 16 | `Exp13_SVM_Sigmoide_C0.1` | Sigmoide | 0.1 | 2151.42 | 984.56 | -0.1147 | 11.1 | — |

---

## Cómo Abrir MLflow UI
Desde la raíz del repositorio:
```powershell
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
Luego abrir **`http://127.0.0.1:5000`** en el navegador:
- En **Experiments → `FineTuning_16_SVM_Kernels_x_C_Retail` → Training runs**: se visualizan las **16 pruebas**, sus gráficas de métricas y sus artefactos.
- En **Experiments → Datasets**: se visualizan los datasets registrados (`FreshRetailNet_50K_sales_history` e `inventory`).
- En **Models** (barra lateral): se visualizan los **5 mejores modelos registrados** (`Mejor_Modelo_SVM_Global` y el mejor de cada kernel).

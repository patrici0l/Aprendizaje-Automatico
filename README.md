# Práctica 2: Máquinas de Soporte Vectorial (SVM) – Fine Tuning y Seguimiento con MLflow

**Dataset:** FreshRetailNet-50K (600 registros, 10 productos) | **Métricas:** RMSE, MAE, $R^2$ y Tiempo de Entrenamiento (ms)

## Estructura en 2 Cuadernos (`.ipynb` y `.html`)

1. **[`Experimentos_MLflow.ipynb`](Experimentos_MLflow.ipynb)** / **[`Experimentos_MLflow.html`](Experimentos_MLflow.html)** (Cuaderno 1 — *Fine Tuning* de SVM y Registro en MLflow):
   - Ejecuta y registra **19 experimentos en MLflow**:
     - **16 experimentos de *Fine Tuning* en SVM** (4 kernels: Lineal, Polinómico, RBF Gaussiano y Sigmoide × 4 configuraciones de hiperparámetros $C, \epsilon, \gamma, d$).
     - **1 experimento de optimización exhaustiva con `GridSearchCV`** (validación cruzada de 5 *folds*, 96 combinaciones = 480 ajustes) sobre el kernel ganador (`Exp19_SVM_GridSearchCV_Polinomico`).
     - **2 experimentos de Random Forest** como línea base comparativa (`Exp17_RandomForest_Base_100` y `Exp18_RandomForest_Optimizado_300`).
   - Guarda todos los modelos, parámetros y figuras (curvas Real vs. Predicho, histogramas de residuos, comparativa general y panel 2×2 de *fine tuning* por kernel) tanto en **MLflow** (`mlflow.db` / `mlruns/`) como en [`modelos/`](modelos) y [`artefactos_mlflow/`](artefactos_mlflow).
   - Exporta automáticamente los modelos ganadores (`mejor_modelo_svm.pkl`, `mejor_modelo_rf.pkl` y `mejor_modelo_global.pkl`).

2. **[`Practica2.ipynb`](Practica2.ipynb)** / **[`Practica2.html`](Practica2.html)** (Cuaderno 2 — Predicción de Nuevos Samples, Explicabilidad, Conclusiones y Referencias APA):
   - Carga los modelos ganadores seleccionados en MLflow (`Exp06_SVM_Polinomico_d2_C5` y `Exp17_RandomForest_Base_100`).
   - Define y valida la función **`predecir_nuevo_sample()`**, evaluándola en **4 casos contrastantes de 4 categorías distintas** (Accesorios, Audio, Computación y Telefonía), en **2 casos adicionales de sensibilidad al descuento** (0 % vs. 20 % de descuento) y en un **pronóstico en lote (*batch*)** para los 10 productos del catálogo.
   - Explica las predicciones mediante *Permutation Feature Importance* (impacto directo de cada variable sobre el RMSE) e incluye **Análisis General, Conclusiones Cuantitativas y Referencias Bibliográficas en formato APA 7.ª edición**.

---

## Tabla de los 19 Experimentos Registrados en MLflow (Ordenados por Mejor RMSE)

| # | Experimento en MLflow | Familia | Kernel | RMSE (u) | MAE (u) | $R^2$ | Tiempo (ms) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **`Exp17_RandomForest_Base_100`** | RandomForest | RandomForest | **537.39** | 309.93 | **0.9305** | 112.0 |
| 2 | `Exp18_RandomForest_Optimizado_300` | RandomForest | RandomForest | 543.08 | 311.12 | 0.9290 | 310.0 |
| 3 | **`Exp06_SVM_Polinomico_d2_C5`** | **SVM** | **Polinómico** | **574.37** | **303.42** | **0.9206** | **26.9** |
| 4 | `Exp19_SVM_GridSearchCV_Polinomico` | SVM | Polinómico (CV-5) | 599.74 | 308.43 | 0.9134 | 6043.5 |
| 5 | `Exp10_SVM_RBF_Gaussiano_C5_g0.02` | SVM | RBF Gaussiano | 609.45 | 313.09 | 0.9106 | 13.1 |
| 6 | `Exp12_SVM_RBF_Gaussiano_C50_g0.01` | SVM | RBF Gaussiano | 619.29 | 309.87 | 0.9076 | 18.1 |
| 7 | `Exp11_SVM_RBF_Gaussiano_C10_g0.05` | SVM | RBF Gaussiano | 647.52 | 318.84 | 0.8990 | 17.5 |
| 8 | `Exp08_SVM_Polinomico_d3_C5` | SVM | Polinómico | 671.65 | 368.92 | 0.8914 | 29.8 |
| 9 | `Exp05_SVM_Polinomico_d2_C1` | SVM | Polinómico | 902.29 | 403.71 | 0.8039 | 13.2 |
| 10 | `Exp07_SVM_Polinomico_d3_C1` | SVM | Polinómico | 941.14 | 450.65 | 0.7867 | 13.7 |
| 11 | `Exp02_SVM_Lineal_C1` | SVM | Lineal | 961.96 | 400.86 | 0.7771 | 16.7 |
| 12 | `Exp03_SVM_Lineal_C10` | SVM | Lineal | 966.39 | 402.75 | 0.7751 | 55.7 |
| 13 | `Exp04_SVM_Lineal_C100` | SVM | Lineal | 967.35 | 402.92 | 0.7746 | 377.3 |
| 14 | `Exp01_SVM_Lineal_C0.1` | SVM | Lineal | 1023.43 | 417.65 | 0.7478 | 17.6 |
| 15 | `Exp16_SVM_Sigmoide_C10_g0.005` | SVM | Sigmoide | 1076.84 | 445.10 | 0.7207 | 12.3 |
| 16 | `Exp09_SVM_RBF_Gaussiano_C1_g0.02` | SVM | RBF Gaussiano | 1103.75 | 472.77 | 0.7066 | 10.8 |
| 17 | `Exp13_SVM_Sigmoide_C0.5_g0.05` | SVM | Sigmoide | 1399.89 | 540.34 | 0.5281 | 15.5 |
| 18 | `Exp14_SVM_Sigmoide_C1_g0.01` | SVM | Sigmoide | 1643.01 | 704.41 | 0.3499 | 10.6 |
| 19 | `Exp15_SVM_Sigmoide_C5_g0.001` | SVM | Sigmoide | 1899.57 | 845.36 | 0.1310 | 11.5 |

---

## Cómo Abrir MLflow UI
Desde la raíz del repositorio:
```powershell
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
Luego abrir **`http://127.0.0.1:5000`** en el navegador.

---

## Estructura del Repositorio
```text
.
├── Experimentos_MLflow.ipynb   # Cuaderno 1: 19 experimentos (16 SVM + GridSearchCV + 2 RF) en MLflow
├── Experimentos_MLflow.html    # Exportación HTML del Cuaderno 1 para entrega en AVAC
├── Practica2.ipynb             # Cuaderno 2: función de predicción, explicabilidad, conclusiones y APA
├── Practica2.html              # Exportación HTML del Cuaderno 2 para entrega en AVAC
├── README.md
├── requirements.txt            # pip install -r requirements.txt
├── data/                       # sales_history.csv (600 filas), inventory.csv (10 productos)
├── modelos/                    # 19 modelos .pkl, modelos ganadores y resumen_experimentos_mlflow.csv
├── artefactos_mlflow/          # 19 gráficos de evaluación + comparativa general + panel 2x2 por kernel
├── mlruns/                     # Artefactos y modelos empaquetados de MLflow
├── mlflow.db                   # Base SQLite de MLflow
└── docs/                       # Guías de práctica (.docx)
```

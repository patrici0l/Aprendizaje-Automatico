# Práctica 2: Experimentos en MLflow y Predicción de Demanda con RF y SVM

**Dataset:** FreshRetailNet-50K (600 registros, 10 productos) | **Métrica:** RMSE, MAE y R²

## Estructura en 2 Cuadernos (`.ipynb`)

1. **[`Experimentos_MLflow.ipynb`](Experimentos_MLflow.ipynb)** (Cuaderno 1 — Solo MLflow y Experimentos):
   - Ejecuta y registra **6 experimentos descriptivos** (4 kernels de SVM + 2 configuraciones de Random Forest) evaluando **RMSE**, MAE y R².
   - Guarda todos los modelos, parámetros y figuras en **MLflow** (`mlflow.db` / `mlruns/`) y en la carpeta local [`modelos/`](modelos).
   - Selecciona y exporta automáticamente el mejor experimento de cada familia (`mejor_modelo_rf.pkl` y `mejor_modelo_svm.pkl`).

2. **[`Practica2.ipynb`](Practica2.ipynb)** (Cuaderno 2 — Solo Predicción y Explicabilidad con RF y SVM):
   - Carga los modelos ganadores seleccionados en MLflow (`Exp6_RandomForest_Optimizado_300` y `Exp2_SVM_Kernel_Polinomico_d2`).
   - Verifica que las predicciones por producto respeten la escala real de demanda sin extrapolaciones fuera de rango.
   - Ejecuta predicciones comparativas con **Random Forest** y **SVM** para nuevos samples individuales y en lote (batch de 10 productos), incluyendo explicabilidad con *Permutation Feature Importance*.

---

## Experimentos Guardados en MLflow (Ordenados por Mejor RMSE)

| Experimento en MLflow | Familia | RMSE (unidades) | MAE (unidades) | R² Score | Selección |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Exp5_RandomForest_Base_100`** | Random Forest | **537.39 u** | **309.93 u** | **0.9305** | **Mejor Experimento Global y Mejor RF** |
| **`Exp6_RandomForest_Optimizado_300`** | Random Forest | 543.08 u | 311.12 u | 0.9290 | RF Profundidad 15 |
| **`Exp2_SVM_Kernel_Polinomico_d2`** | SVM | **574.37 u** | **303.42 u** | **0.9206** | **Mejor Experimento SVM** |
| **`Exp3_SVM_Kernel_RBF_Gaussiano`** | SVM | 609.45 u | 313.09 u | 0.9106 | Kernel Radial Gaussiano |
| **`Exp1_SVM_Kernel_Lineal`** | SVM | 961.96 u | 400.86 u | 0.7771 | Kernel Lineal |
| **`Exp4_SVM_Kernel_Sigmoide`** | SVM | 1399.89 u | 540.34 u | 0.5281 | Kernel Sigmoide |

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
├── Experimentos_MLflow.ipynb   # Cuaderno 1: experimentos + registro en MLflow
├── Practica2.ipynb             # Cuaderno 2: predicción y explicabilidad
├── README.md
├── requirements.txt            # pip install -r requirements.txt
├── data/                       # sales_history.csv, inventory.csv
├── modelos/                    # Modelos .pkl y resumen de experimentos
├── artefactos_mlflow/          # Gráficos de evaluación por experimento
├── mlruns/                     # Artefactos y modelos de MLflow
├── mlflow.db                   # Base SQLite de MLflow
└── docs/                       # Guía de la práctica (.docx)
```
> Ejecutar los notebooks desde la raíz del repositorio para que las rutas relativas funcionen.

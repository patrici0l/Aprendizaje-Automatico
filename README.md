# Práctica 2: Máquinas de Soporte Vectorial (SVR) – Fine Tuning (24 Combinaciones) y Seguimiento con MLflow

**Asignatura:** Aprendizaje Automático | **Docente:** Ing. Remigio Hurtado, PhD.  
**Dataset:** FreshRetailNet-50K (`data/sales_history.csv`: 600 registros, 10 productos × 60 días del `2024-03-28` al `2024-05-26`; `data/inventory.csv`: catálogo de 10 productos)  
**Experimento en MLflow:** `SVR_Prediccion_Demanda_Retail`

---

## Estructura en 2 Cuadernos (`.ipynb` y `.html`)

1. **[`Experimentos_MLflow.ipynb`](Experimentos_MLflow.ipynb)** / **[`Experimentos_MLflow.html`](Experimentos_MLflow.html)** (Cuaderno 1 — Limpieza, *Fine Tuning* de 24 Combinaciones y Registro en MLflow):
   - **Punto 1:** Configuración del servidor MLflow con base SQLite (`mlflow.db`) y almacén de artefactos (`mlruns/`).
   - **Punto 2:** Limpieza de datos (0 nulos, 0 duplicados), extracción de variables de calendario desde `fecha_venta` (`dia_semana`, `es_fin_de_semana`), `StandardScaler` (6 variables numéricas), `OneHotEncoder` (2 categóricas) y partición estratificada 80/20 (480 entrenamiento / 120 prueba).
   - **Punto 3:** Entrenamiento y validación cruzada (5 *folds*) de **24 combinaciones** (**4 kernels**: `linear`, `poly`, `rbf`, `sigmoid` × **6 valores de $C \in \{0.1, 0.5, 1.0, 5.0, 10.0, 100.0\}$**), registrando las 24 corridas (`Rank_01` a `Rank_24`) con 6 métricas (`RMSE_CV`, `STD_CV`, `RMSE_Train_CV`, `RMSE_Test`, `MAE_Test`, `R2_Test`) y guardando los **4 mejores modelos en la pestaña `Models`** de MLflow.
   - **Punto 4:** Gráficas comparativas de las 24 combinaciones y panel 2×2 por tipo de kernel.
   - **Punto 5:** Exportación a `modelos/` de los 4 mejores modelos por kernel, el Mejor Modelo Global (`Rank_01_SVR_poly_C5.0`) y el Peor Modelo Global (`Rank_24_SVR_sigmoid_C0.1`).
   - *Cada punto incluye al final su propio bloque de **Conclusión y Recomendación**, más la sección general de Conclusiones, Recomendaciones y Referencias APA.*

2. **[`Practica2.ipynb`](Practica2.ipynb)** / **[`Practica2.html`](Practica2.html)** (Cuaderno 2 — Predicción por Producto y Fecha, Mejor vs. Peor Modelo, Nuevos Samples y Explicabilidad):
   - **Punto 1:** Carga de los 4 mejores modelos por kernel y del Peor Modelo Global desde `modelos/`.
   - **Punto 2:** Carga del dataset limpio y catálogo de los 10 productos.
   - **Punto 3 (Mejor Modelo):** Predicción por producto y fecha y gráfica de **2 ítems específicos (`ID 4` e `ID 70`) con el MEJOR MODELO (`Rank_01_SVR_poly_C5.0`)** frente al Valor Real (`Error: 0.23%` y `0.14%`).
   - **Punto 4 (Peor Modelo):** Predicción por producto y fecha y gráfica de los **mismos 2 ítems específicos con el PEOR MODELO (`Rank_24_SVR_sigmoid_C0.1`)** frente al Valor Real (`Error: 86.34%` y `89.37%`).
   - **Punto 5:** Auditoría promedio por producto en el conjunto de prueba (Tabla + Gráfica).
   - **Punto 6:** Función `predecir_nuevo_sample()` evaluada en 4 casos de distintas categorías + análisis de sensibilidad al descuento (Tabla + Gráfica).
   - **Punto 7:** Explicabilidad de las predicciones con *Permutation Feature Importance* comparando el Mejor, 2.° Mejor y Peor Modelo (Tabla + Gráfica).
   - **Punto 8:** Matriz resumen de las 24 combinaciones, Conclusiones detalladas por punto, Recomendaciones por punto y Referencias APA 7.ª edición.

---

## Tabla de las 24 Combinaciones Registradas en MLflow (Ordenadas de `Rank_01` a `Rank_24`)

| Rank | Experimento en MLflow | Kernel | $C$ | `RMSE_Test` (u) | `MAE_Test` (u) | `R2_Test` | `RMSE_CV` (u) | `STD_CV` (u) | Registro en Pestaña `Models` |
| :-: | :--- | :---: | :-: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`Rank_01_SVR_poly_C5.0`** | **poly ($d=2$)** | **5.0** | **574.37** | **303.42** | **0.9206** | 901.21 | 170.39 | **`Top1_SVR_poly_C5.0` (Mejor Global)** |
| **2** | **`Rank_02_SVR_poly_C10.0`** | **poly ($d=2$)** | **10.0** | **599.74** | **308.43** | **0.9134** | 872.61 | 132.68 | **`Top2_SVR_poly_C10.0`** |
| **3** | **`Rank_03_SVR_rbf_C5.0`** | **rbf** | **5.0** | **609.45** | **313.09** | **0.9106** | 888.75 | 129.27 | **`Top3_SVR_rbf_C5.0` (Mejor RBF)** |
| **4** | **`Rank_04_SVR_rbf_C100.0`** | **rbf** | **100.0** | **616.89** | **319.52** | **0.9084** | 821.44 | 91.92 | **`Top4_SVR_rbf_C100.0`** |
| 5 | `Rank_05_SVR_rbf_C10.0` | rbf | 10.0 | 635.56 | 320.56 | 0.9027 | 846.93 | 106.13 | — |
| 6 | `Rank_06_SVR_poly_C100.0` | poly ($d=2$) | 100.0 | 666.84 | 318.57 | 0.8929 | 987.27 | 183.41 | — |
| 7 | `Rank_07_SVR_poly_C1.0` | poly ($d=2$) | 1.0 | 902.29 | 403.71 | 0.8039 | 1348.86 | 246.30 | — |
| 8 | `Rank_08_SVR_linear_C5.0` | linear | 5.0 | 953.35 | 400.20 | 0.7811 | 1361.16 | 283.85 | Mejor Lineal |
| 9 | `Rank_09_SVR_linear_C1.0` | linear | 1.0 | 961.96 | 400.86 | 0.7771 | 1368.17 | 300.42 | — |
| 10 | `Rank_10_SVR_linear_C10.0` | linear | 10.0 | 966.39 | 402.75 | 0.7751 | 1364.82 | 287.41 | — |
| 11 | `Rank_11_SVR_linear_C100.0` | linear | 100.0 | 967.35 | 402.92 | 0.7746 | 1359.93 | 284.72 | — |
| 12 | `Rank_12_SVR_linear_C0.5` | linear | 0.5 | 992.55 | 401.99 | 0.7628 | 1364.42 | 311.47 | — |
| 13 | `Rank_13_SVR_linear_C0.1` | linear | 0.1 | 1023.43 | 417.65 | 0.7478 | 1425.91 | 387.26 | — |
| 14 | `Rank_14_SVR_sigmoid_C100.0` | sigmoid | 100.0 | 1055.43 | 419.40 | 0.7317 | 1428.50 | 343.20 | Mejor Sigmoide |
| 15 | `Rank_15_SVR_sigmoid_C10.0` | sigmoid | 10.0 | 1076.84 | 445.10 | 0.7207 | 1535.60 | 406.14 | — |
| 16 | `Rank_16_SVR_rbf_C1.0` | rbf | 1.0 | 1103.75 | 472.77 | 0.7066 | 1630.97 | 460.77 | — |
| 17 | `Rank_17_SVR_sigmoid_C5.0` | sigmoid | 5.0 | 1265.99 | 510.19 | 0.6140 | 1756.07 | 455.81 | — |
| 18 | `Rank_18_SVR_poly_C0.5` | poly ($d=2$) | 0.5 | 1278.09 | 565.30 | 0.6066 | 1585.94 | 321.41 | — |
| 19 | `Rank_19_SVR_rbf_C0.5` | rbf | 0.5 | 1450.44 | 599.66 | 0.4934 | 2009.19 | 480.00 | — |
| 20 | `Rank_20_SVR_poly_C0.1` | poly ($d=2$) | 0.1 | 1703.52 | 816.38 | 0.3011 | 2176.55 | 346.40 | — |
| 21 | `Rank_21_SVR_sigmoid_C1.0` | sigmoid | 1.0 | 1899.86 | 845.47 | 0.1308 | 2371.96 | 497.44 | — |
| 22 | `Rank_22_SVR_rbf_C0.1` | rbf | 0.1 | 2020.40 | 902.76 | 0.0170 | 2486.09 | 509.97 | — |
| 23 | `Rank_23_SVR_sigmoid_C0.5` | sigmoid | 0.5 | 2048.14 | 924.22 | -0.0102 | 2508.51 | 509.38 | — |
| **24** | **`Rank_24_SVR_sigmoid_C0.1`** | **sigmoid** | **0.1** | **2151.42** | **984.56** | **-0.1147** | 2598.82 | 515.76 | **Peor Modelo Global** |

---

## Cómo Abrir MLflow UI
Ejecutar desde la raíz del repositorio:
```powershell
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
Abrir **`http://127.0.0.1:5000`** y seleccionar **`Model training`** arriba a la izquierda:
- **`Runs` (vista gráfica `📈`):** muestra las **24 combinaciones** en las 6 gráficas de métricas.
- **`Models` (vista gráfica `📈`):** muestra únicamente los **4 mejores modelos** con su nombre descriptivo (`Rank_01` a `Rank_04`).
- **`Runs` → seleccionar las 24 → `Compare`:** muestra el *Parallel Coordinates Plot* (`C` → `kernel` → `ranking_cv` → `RMSE_CV`).

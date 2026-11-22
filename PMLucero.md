# Guía Rápida de Exposición y Sustentación — Patricio Lucero (`PMLucero`)

**Tema:** Máquinas de Soporte Vectorial para Regresión (SVR) – *Fine Tuning* (24 Combinaciones) y Seguimiento con MLflow  
**Dataset:** `FreshRetailNet-50K` (`data/sales_history.csv`: 600 registros = 10 productos × 60 días del `2024-03-28` al `2024-05-26`)

---

## PARTE 1: Cuaderno 1 (`Experimentos_MLflow.ipynb`) + Interfaz de MLflow

### Punto 1. Configuración de MLflow
- **Qué se hizo:** Se conectó MLflow a una base de datos SQLite local (`mlflow.db`) y a la carpeta `mlruns/` bajo el experimento `SVR_Prediccion_Demanda_Retail`.
- **Qué decir:** *"Usamos SQLite como backend de MLflow para poder registrar no solo las corridas de entrenamiento (`Runs`), sino también publicar los mejores modelos en el registro oficial (`Models` y `Model registry`)."*

### Punto 2. Limpieza de Datos, Ingeniería de Variables y Preprocesamiento
- **Qué se hizo:**
  1. Verificación de **0 nulos y 0 duplicados** en las 600 filas.
  2. Extracción de variables de calendario desde `fecha_venta`: `dia_semana` (0=Lun a 6=Dom) y `es_fin_de_semana` (0 o 1).
  3. Preprocesador `ColumnTransformer`:
     - **6 variables numéricas** (`stock_en_momento`, `discount`, `temperatura`, `dia_semana`, `es_fin_de_semana`, `holiday_flag`) escaladas con **`StandardScaler`**.
     - **2 variables categóricas** (`product_id`, `categoria`) codificadas con **`OneHotEncoder`**.
  4. Transformación del objetivo con **`TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`** para estabilizar las diferencias de escala entre productos (desde 31 u/día en Laptops hasta 6272 u/día en Cables HDMI).
  5. División **80 % entrenamiento (480 filas)** y **20 % prueba (120 filas)** estratificada por `product_id`.

### Punto 3. *Fine Tuning* de las 24 Combinaciones (4 Kernels × 6 Valores de $C$)
- **Qué se hizo:** Se cruzaron los **4 kernels de SVR** (`linear`, `poly` grado 2, `rbf`, `sigmoid`) con **6 valores de regularización $C \in \{0.1, 0.5, 1.0, 5.0, 10.0, 100.0\}$** = **24 combinaciones**.
- **Métricas registradas en cada una de las 24 corridas:**
  - En validación cruzada de 5 *folds* (entrenamiento): `RMSE_CV`, `STD_CV`, `RMSE_Train_CV`.
  - En conjunto de prueba (test): `RMSE_Test`, `MAE_Test`, `R2_Test`.
- **Resultado clave:**
  - **Mejor Modelo Global (`Rank_01_SVR_poly_C5.0`):** Kernel Polinómico ($d=2, C=5.0$) $\rightarrow$ **`RMSE_Test = 574.37 u`**, **`MAE_Test = 303.42 u`**, **`R2_Test = 0.9206`** (explica el 92.06 % de la varianza).
  - **Peor Modelo Global (`Rank_24_SVR_sigmoid_C0.1`):** Kernel Sigmoide ($C=0.1$) $\rightarrow$ **`RMSE_Test = 2151.42 u`**, **`MAE_Test = 984.56 u`**, **`R2_Test = -0.1147`**.
  - El *Fine Tuning* redujo el error RMSE en un **73.3 %**.

### Punto 4 y 5. Gráficas Comparativas y Registro de los Mejores Modelos en `Models`
- **Qué mostrar en MLflow UI (`http://127.0.0.1:5000`, modo `Model training`):**
  1. **Pestaña `Runs` (ícono de gráfica `📈`):** están las **24 barras** (`Rank_01` a `Rank_24`) en las 6 métricas.
  2. **Pestaña `Models` (ícono de gráfica `📈`):** se registraron únicamente los **4 mejores modelos** con su nombre propio (`Rank_01_SVR_poly_C5.0`, `Rank_02_SVR_poly_C10.0`, `Rank_03_SVR_rbf_C5.0`, `Rank_04_SVR_rbf_C100.0`).
  3. **Botón `Compare` (*Parallel Coordinates Plot*):** conecta las 24 combinaciones entre `C` $\rightarrow$ `kernel` $\rightarrow$ `ranking_cv` $\rightarrow$ `RMSE_CV`.

---

## PARTE 2: Cuaderno 2 (`Practica2.ipynb`) — Predicciones y Explicabilidad

### Punto 1 y 2. Carga de Modelos Seleccionados y Catálogo
- **Qué se hizo:** Se cargan desde `modelos/` los 4 mejores modelos por kernel (`Polinomico C=5`, `RBF C=5`, `Lineal C=5`, `Sigmoide C=100`) y el **Peor Modelo Global (`Sigmoide C=0.1`)**, junto con el catálogo de 10 productos.

### Punto 3. Predicción y Gráfica de 2 Ítems Específicos con el MEJOR MODELO (`Rank_01_SVR_poly_C5.0`)
- **Qué se hizo:** La función `graficar_dos_items_modelo()` recibe 2 ítems específicos con su fecha `(product_id, fecha_venta)`, busca el registro real en el dataset y grafica los 60 días de historial + barras comparativas **Valor Real vs. Mejor Modelo**:
  - **Ítem 1 (`ID 4 - Cable HDMI 4K`, fecha `2024-04-08`):** Valor Real = **`3000 u`** | Predicción Mejor SVR = **`3007.0 u`** | **Error = `0.23%`**.
  - **Ítem 2 (`ID 70 - Monitor LG 24"`, fecha `2024-05-18`):** Valor Real = **`4037 u`** | Predicción Mejor SVR = **`4042.8 u`** | **Error = `0.14%`**.
- **Si el docente pide cambiar de producto o fecha en vivo:** En la última línea de la celda del Punto 3 cambias `item_1=(ID, "YYYY-MM-DD")` (IDs válidos: `4, 18, 19, 21, 38, 70, 72, 117, 190, 194`; fechas entre `2024-03-28` y `2024-05-26`) y ejecutas la celda.

### Punto 4. Predicción y Gráfica de los Mismos 2 Ítems con el PEOR MODELO (`Rank_24_SVR_sigmoid_C0.1`)
- **Qué se hizo:** Justo debajo, en su propio apartado, se evalúan los mismos 2 ítems y fechas con el **Peor Modelo (`sigmoid C=0.1`)**:
  - **Ítem 1 (`ID 4`, `2024-04-08`):** Valor Real = **`3000 u`** | Predicción Peor SVR = **`409.8 u`** | **Error = `86.34%`**.
  - **Ítem 2 (`ID 70`, `2024-05-18`):** Valor Real = **`4037 u`** | Predicción Peor SVR = **`429.0 u`** | **Error = `89.37%`**.
- **Qué decir:** *"En la gráfica se ve por qué el peor modelo tiene $R^2$ negativo: como el kernel sigmoide con $C=0.1$ está fuertemente subajustado, predice una línea plana de ~410 a 430 unidades para cualquier producto y fecha, mientras que el mejor modelo polinómico con $C=5.0$ sigue con precisión los picos reales."*

### Punto 5. Auditoría Promedio por Producto en el Conjunto de Prueba (10 Productos)
- **Qué se hizo:** Tabla y gráfica comparando la demanda real promedio en las 120 muestras de test frente a los modelos en los 10 productos (desde la Laptop Gamer con `25.8 u` reales $\rightarrow$ `26.7 u` en el Mejor SVR, hasta el Cable HDMI con `5510.0 u` reales $\rightarrow$ `5278.9 u` en el Mejor SVR).

### Punto 6. Función `predecir_nuevo_sample()` (4 Casos de Distintas Categorías + Sensibilidad al Descuento)
- **Qué se hizo:** Función para predecir nuevos escenarios que no están en una fecha histórica, probada en **4 casos de categorías distintas** (*Accesorios*, *Audio*, *Computación*, *Telefonía*) verificando si el stock alcanza (`OK: Stock Suficiente` o `ALERTA: Reabastecer`).
- **Sensibilidad al descuento (Producto ID 38 - Parlante Bluetooth):** Al aplicar un 20 % de descuento (`discount` de `1.00` a `0.80`), el Mejor SVR incrementa la demanda estimada en **+81.3 %** (`152.5 u` $\rightarrow$ `276.5 u`), capturando la elasticidad comercial.

### Punto 7. Explicabilidad (*Permutation Feature Importance*)
- **Qué se hizo:** Se permutó cada variable 10 veces en el conjunto de prueba midiendo cuánto sube el error RMSE:
  1. `product_id` (**+1384.39 u**) y `categoria` (**+1152.72 u**): definen la escala base del producto.
  2. `holiday_flag` (**+1070.91 u**) y `discount` (**+1051.87 u**): son los dos factores externos que más disparan las ventas.
  3. `es_fin_de_semana` (**+124.99 u**), `dia_semana` (**+114.37 u**), `stock_en_momento` (**+90.92 u**) y `temperatura` (**+59.86 u**).
  - En el Peor Modelo todas las variables dan importancia cercana a `0 u`, confirmando que no aprendió patrones.

### Punto 8. Conclusiones, Recomendaciones y Referencias APA
-Cada punto tiene su conclusión y recomendación debajo de su gráfica, y en el Punto 8 se consolida la matriz de las 24 combinaciones junto con las 7 referencias en formato **APA 7.ª edición** (Breiman, Cortes & Vapnik, Dingdong Limited, Pedregosa/Scikit-learn, Raschka, Smola & Schölkopf, Zaharia/MLflow).

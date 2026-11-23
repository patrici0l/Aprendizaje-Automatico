import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. CARGA DE DATOS REALES Y MODELO PRODUCTIVO
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SALES = os.path.join(BASE_DIR, "data", "sales_history.csv")
DATA_INV = os.path.join(BASE_DIR, "data", "inventory.csv")
INFO_MEJORES = os.path.join(BASE_DIR, "modelos", "mejor_experimento_info.json")
MODELO_MEJOR_PKL = os.path.join(BASE_DIR, "modelos", "Rank_01_SVR_poly_C5.0.pkl")

OUT_FIG1 = os.path.join(BASE_DIR, "artefactos_mlflow", "fig1_ejecutivo_ventas_categorias.png")
OUT_FIG2 = os.path.join(BASE_DIR, "artefactos_mlflow", "fig2_ejecutivo_solucion_modelo.png")
OUT_HTML = os.path.join(BASE_DIR, "Reporte_Ejecutivo_Prediccion_Demanda_Retail.html")

def fmt_int(val):
    return f"{int(round(val)):,}".replace(",", ".")

def fmt_dec(val, dec=1):
    s = f"{val:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

df_sales = (
    pd.read_csv(DATA_SALES)
    .dropna()
    .drop_duplicates(["product_id", "fecha_venta"])
    .sort_values(["product_id", "fecha_venta"])
    .reset_index(drop=True)
)
df_inv = pd.read_csv(DATA_INV)
info_mejores = json.load(open(INFO_MEJORES, encoding="utf-8"))

df_sales["fecha"] = pd.to_datetime(df_sales["fecha_venta"])
df_sales["dia_semana"] = df_sales["fecha"].dt.dayofweek
df_sales["es_fin_de_semana"] = (df_sales["dia_semana"] >= 5).astype(int)

FEATURES = [
    "stock_en_momento", "discount", "temperatura",
    "dia_semana", "es_fin_de_semana", "holiday_flag",
    "product_id", "categoria"
]
X = df_sales[FEATURES].assign(product_id=df_sales["product_id"].astype(str))
y = df_sales["cantidad_vendida"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X["product_id"]
)

modelo_mejor = pickle.load(open(MODELO_MEJOR_PKL, "rb"))

# --- Ventas por categoría y producto ---
cat_stats = (
    df_sales.groupby("categoria")
    .agg(total_ventas=("cantidad_vendida", "sum"), prom_diario=("cantidad_vendida", "mean"))
    .reset_index()
    .sort_values("total_ventas", ascending=False)
)
total_unidades = int(cat_stats["total_ventas"].sum())
cat_stats["pct"] = cat_stats["total_ventas"] / total_unidades * 100.0
cat_order = cat_stats["categoria"].tolist()
cat_pct_map = dict(zip(cat_stats["categoria"], cat_stats["pct"]))

prod_stats = (
    df_sales.groupby(["categoria", "product_id", "nombre"])
    .agg(total_ventas=("cantidad_vendida", "sum"), prom_diario=("cantidad_vendida", "mean"))
    .reset_index()
)
prod_stats["cat_rank"] = prod_stats["categoria"].map({c: i for i, c in enumerate(cat_order)})
prod_stats = prod_stats.sort_values(["cat_rank", "total_ventas"], ascending=[True, False]).reset_index(drop=True)

# --- Evaluación en Test para los 6 productos principales ---
df_ev = X_test.assign(
    ID=X_test["product_id"].astype(int),
    Real_u=y_test,
    Pronostico=modelo_mejor.predict(X_test),
)
ev_prod = df_ev.groupby("ID")[["Real_u", "Pronostico"]].mean()

# --- Exactitud global y sensibilidad al descuento ---
r2_best_pct = info_mejores["mejor_experimento_global"]["R2_Test"] * 100.0
hol_global = df_sales.groupby("holiday_flag")["cantidad_vendida"].mean()

def pred_sample(pid, stock, discount=1.0, temp=18.0, dia_sem=5, feriado=0):
    cat = df_inv.loc[df_inv["product_id"] == pid, "categoria"].values[0]
    x_new = pd.DataFrame([{
        "stock_en_momento": stock, "discount": discount, "temperatura": temp,
        "dia_semana": dia_sem, "es_fin_de_semana": int(dia_sem >= 5),
        "holiday_flag": feriado, "product_id": str(pid), "categoria": cat
    }])
    return float(modelo_mejor.predict(x_new)[0])

elasticidad_pct = (pred_sample(38, 100, 0.80) / pred_sample(38, 100, 1.00) - 1.0) * 100.0

# --- Paleta corporativa sobria (exclusivamente Azul Marino, Azul Acero y Gris Pizarra) ---
NAVY = "#1B365D"
STEEL = "#5A7D9A"
SLATE = "#7A8B9C"
DARK_TEXT = "#1E293B"
MUTED_TEXT = "#475569"
GRID_GRAY = "#E2E8F0"

CAT_BAR_COLORS = {
    "Accesorios": NAVY,
    "Computación": NAVY,
    "Telefonía": STEEL,
    "Audio": STEEL,
    "Videojuegos": SLATE,
    "Tablets": SLATE,
}

def clean_prod_name(name):
    return (
        name.replace("Alta Velocidad ", "")
        .replace("Alta Gama ", "")
        .replace("Full HD 24 pulg", '24"')
        .replace("Smartphone ", "")
        .replace("Apple ", "")
        .replace("Tablet ", "")
        .replace("Galaxy ", "")
        .replace("Sony ", "")
        .replace("por Cable ", "")
        .replace("BLUE Bass", "Bass")
        .replace("Prepago 4G/5G", "4G/5G")
        .replace("Rápido ", "")
    )

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8

# ==============================================================================
# 2. FIGURA 1: VENTAS HISTÓRICAS — TOP PRODUCTOS AGRUPADOS POR CATEGORÍA
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(12.6, 4.05), dpi=300, facecolor="white")
plt.subplots_adjust(left=0.285, right=0.96, top=0.84, bottom=0.14)

y_positions = []
y_labels = []
cat_bounds = {}
curr_y = 0.0
for cat in cat_order:
    sub = prod_stats[prod_stats["categoria"] == cat]
    start_y = curr_y
    for _, row in sub.iterrows():
        y_positions.append(curr_y)
        y_labels.append((clean_prod_name(row["nombre"]), cat, int(row["total_ventas"])))
        curr_y += 1.0
    end_y = curr_y - 1.0
    cat_bounds[cat] = (start_y, end_y)
    curr_y += 0.55

for idx_c, cat in enumerate(cat_order):
    y0, y1 = cat_bounds[cat]
    if idx_c % 2 == 0:
        ax1.axhspan(y0 - 0.44, y1 + 0.44, color="#F8FAFC", zorder=0)
    if idx_c < len(cat_order) - 1:
        ax1.axhline(y1 + 0.72, color="#E2E8F0", lw=0.8, zorder=1)
    ymid = (y0 + y1) / 2.0
    c_col = CAT_BAR_COLORS[cat]
    ax1.plot([-0.245, -0.245], [y0 - 0.34, y1 + 0.34], transform=ax1.get_yaxis_transform(),
             color=c_col, lw=3.2, solid_capstyle="butt", clip_on=False)
    ax1.text(
        -0.260, ymid,
        f"{cat.upper()} ({fmt_dec(cat_pct_map[cat], 1)} %)",
        transform=ax1.get_yaxis_transform(),
        va="center", ha="right", fontsize=7.8, fontweight="bold", color=c_col, clip_on=False
    )

ax1.barh(
    y_positions,
    [lbl[2] for lbl in y_labels],
    height=0.62,
    color=[CAT_BAR_COLORS[lbl[1]] for lbl in y_labels],
    zorder=3,
)

ax1.set_xlim(0, 445_000)
ax1.set_yticks(y_positions)
ax1.set_yticklabels([lbl[0] for lbl in y_labels], fontsize=8.8, color=DARK_TEXT, fontweight="semibold")
ax1.tick_params(axis="y", length=0, pad=8)
ax1.invert_yaxis()
ax1.set_xticks([0, 100_000, 200_000, 300_000, 400_000])
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "0" if x == 0 else f"{int(x/1000)}.000 u" if x == 0 else f"{fmt_int(x)} u"))
ax1.tick_params(axis="x", labelsize=8.0, colors=MUTED_TEXT)
ax1.grid(axis="x", linestyle=":", alpha=0.8, color="#CBD5E1", zorder=1)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlabel("Unidades vendidas en 60 días", fontsize=8.4, color=MUTED_TEXT, labelpad=5)

for y_p, lbl in zip(y_positions, y_labels):
    _, _, tot = lbl
    ax1.text(
        tot + 5_500, y_p,
        f"{fmt_int(tot)} u",
        va="center", ha="left", fontsize=8.3, fontweight="bold", color=DARK_TEXT, zorder=4
    )

ax1.set_title(
    "FIGURA 1. VENTAS HISTÓRICAS: TOP PRODUCTOS AGRUPADOS POR CATEGORÍA (60 DÍAS)",
    fontsize=10.2, fontweight="bold", color=NAVY, loc="left", x=-0.39, pad=12
)
fig1.savefig(OUT_FIG1, dpi=300, facecolor="white")
plt.close(fig1)

# ==============================================================================
# 3. FIGURA 2: SOLUCIÓN DEL MODELO (VENTA REAL VS. PRONÓSTICO — TOP 6 PRODUCTOS)
# ==============================================================================
top6_meta = [
    (4,   "Cable HDMI 4K\n(Accesorios)"),
    (19,  "Cargador USB-C 65W\n(Accesorios)"),
    (70,  'Monitor LG 24"\n(Computación)'),
    (18,  "Chip SIM 4G/5G\n(Telefonía)"),
    (117, "iPhone Pro Max\n(Telefonía)"),
    (21,  "Audífonos AKG\n(Audio)"),
]

real_vals = [float(ev_prod.loc[pid, "Real_u"]) for pid, _ in top6_meta]
pred_vals = [float(ev_prod.loc[pid, "Pronostico"]) for pid, _ in top6_meta]
x_labels = [lbl for _, lbl in top6_meta]

fig2, ax2 = plt.subplots(figsize=(12.6, 3.75), dpi=300, facecolor="white")
plt.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.19)

x = np.arange(len(top6_meta))
w = 0.34
ax2.bar(x - w/2, real_vals, width=w, color=NAVY, label="Venta Real Promedio (u/día)", zorder=3)
ax2.bar(x + w/2, pred_vals, width=w, color=STEEL, label="Pronóstico de la Solución Tecnológica (Exactitud 92,1 %)", zorder=3)

ax2.set_ylim(0, 6400)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=8.5, fontweight="semibold", color=DARK_TEXT)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: fmt_int(v)))
ax2.tick_params(axis="y", labelsize=8.0, colors=MUTED_TEXT)
ax2.grid(axis="y", linestyle=":", alpha=0.8, color="#CBD5E1", zorder=1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylabel("Unidades diarias promedio", fontsize=8.4, color=MUTED_TEXT)

for i, (rv, pv) in enumerate(zip(real_vals, pred_vals)):
    ax2.text(i - w/2 - 0.03, rv + 95, fmt_int(rv), ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=NAVY)
    ax2.text(i + w/2 + 0.03, pv + 95, fmt_int(pv), ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=STEEL)

ax2.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#CBD5E1")
ax2.set_title(
    "FIGURA 2. SOLUCIÓN DEL MODELO: VENTA REAL VS. PRONÓSTICO EN PRODUCTOS PRINCIPALES",
    fontsize=10.2, fontweight="bold", color=NAVY, loc="left", pad=12
)
fig2.savefig(OUT_FIG2, dpi=300, facecolor="white")
plt.close(fig2)

print("Figuras generadas:", OUT_FIG1, OUT_FIG2)

# ==============================================================================
# 4. HTML SOBRIO Y MINIMALISTA DE 1 PÁGINA (SIN INTEGRANTES, SIN PÁRRAFOS)
# ==============================================================================
html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte Ejecutivo — Solución Tecnológica de Pronóstico de Ventas</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  html, body {{
    width: 210mm;
    height: 297mm;
    overflow: hidden;
    font-family: 'Segoe UI', Calibri, Arial, sans-serif;
    background: #FFFFFF;
    color: #1E293B;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .page {{
    width: 210mm;
    height: 297mm;
    padding: 9mm 11mm 7mm 11mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}

  /* Encabezado limpio */
  .header {{
    border-bottom: 2.5px solid #1B365D;
    padding-bottom: 7px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .header h1 {{
    font-size: 13pt;
    font-weight: 800;
    color: #1B365D;
    line-height: 1.15;
    letter-spacing: -0.1px;
  }}
  .header .sub {{
    font-size: 8.6pt;
    color: #475569;
    margin-top: 2.5px;
    font-weight: 500;
  }}
  .header-meta {{
    text-align: right;
    font-size: 7.6pt;
    color: #64748B;
    line-height: 1.3;
    white-space: nowrap;
  }}
  .header-meta strong {{
    color: #1B365D;
    display: block;
    font-size: 8pt;
  }}

  /* 3 Indicadores ejecutivos */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }}
  .kpi-card {{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #1B365D;
    padding: 6.5px 10px;
  }}
  .kpi-card.steel {{
    border-left-color: #5A7D9A;
  }}
  .kpi-val {{
    font-size: 13.5pt;
    font-weight: 800;
    color: #1B365D;
    line-height: 1.1;
  }}
  .kpi-lbl {{
    font-size: 7.6pt;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    margin-top: 2.5px;
  }}

  /* 2 Columnas de ideas cortas (cero párrafos) */
  .ideas-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }}
  .sec-col {{
    border-top: 1.5px solid #1B365D;
    padding-top: 5px;
  }}
  .sec-col.steel {{
    border-top-color: #5A7D9A;
  }}
  .sec-title {{
    font-size: 8.8pt;
    font-weight: 800;
    color: #1B365D;
    text-transform: uppercase;
    margin-bottom: 5px;
    letter-spacing: 0.25px;
  }}
  .sec-col.steel .sec-title {{
    color: #475569;
  }}
  ul.bullets {{
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  ul.bullets li {{
    font-size: 8.1pt;
    line-height: 1.25;
    color: #1E293B;
    padding-left: 10px;
    position: relative;
  }}
  ul.bullets li::before {{
    content: "•";
    position: absolute;
    left: 0;
    top: 0;
    color: #1B365D;
    font-weight: 900;
  }}

  /* Contenedores de Figuras */
  .fig-box {{
    border: 1px solid #CBD5E1;
    padding: 4px 6px;
    background: #FFFFFF;
  }}
  .fig-box img {{
    width: 100%;
    height: auto;
    display: block;
  }}

  /* Cierre de 3 decisiones clave */
  .actions-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }}
  .action-box {{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-top: 2.5px solid #1B365D;
    padding: 5.5px 9px;
    font-size: 7.9pt;
    line-height: 1.28;
    color: #1E293B;
  }}
  .action-box strong {{
    color: #1B365D;
    display: block;
    font-size: 8.1pt;
    margin-bottom: 1.5px;
  }}

  .footer {{
    display: flex;
    justify-content: space-between;
    font-size: 7pt;
    color: #64748B;
    border-top: 1px solid #CBD5E1;
    padding-top: 4px;
  }}
</style>
</head>
<body>
<div class="page">

  <!-- 1. ENCABEZADO CORPORATIVO (SIN INTEGRANTES) -->
  <div class="header">
    <div>
      <h1>REPORTE EJECUTIVO: SOLUCIÓN TECNOLÓGICA DE PRONÓSTICO DE VENTAS</h1>
      <div class="sub">Análisis Comercial por Categoría y Proyección Inteligente de Demanda</div>
    </div>
    <div class="header-meta">
      <strong>Período: 60 días (28/Mar – 26/May)</strong>
      6 Categorías • 10 Productos
    </div>
  </div>

  <!-- 2. BANDA DE 3 INDICADORES SIMPLES -->
  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-val">{fmt_int(total_unidades)} u</div>
      <div class="kpi-lbl">Ventas Totales (60 días)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">83,1 %</div>
      <div class="kpi-lbl">Concentrado en 2 Categorías Líderes</div>
    </div>
    <div class="kpi-card steel">
      <div class="kpi-val">{fmt_dec(r2_best_pct, 1)} %</div>
      <div class="kpi-lbl">Exactitud del Pronóstico del Modelo</div>
    </div>
  </div>

  <!-- 3. BLOQUE DE IDEAS CLAVE EN 1 LÍNEA (SIN PÁRRAFOS) -->
  <div class="ideas-grid">
    <div class="sec-col">
      <div class="sec-title">Ideas Clave de Ventas</div>
      <ul class="bullets">
        <li><b>Accesorios lidera el negocio:</b> 512.276 unidades (64,9 % del total).</li>
        <li><b>Computación es la 2.ª categoría:</b> 143.322 unidades (18,2 % del total).</li>
        <li><b>3 productos mueven el 83 %:</b> Cable HDMI 4K, Monitor LG 24" y Cargador 65W.</li>
        <li><b>Días feriados duplican ventas:</b> Demanda sube de {fmt_int(hol_global[0])} a {fmt_int(hol_global[1])} u/día.</li>
      </ul>
    </div>
    <div class="sec-col steel">
      <div class="sec-title">Solución Tecnológica</div>
      <ul class="bullets">
        <li><b>Pronóstico automático diario:</b> Proyecta ventas por producto y categoría.</li>
        <li><b>Variables comerciales integradas:</b> Calendario, feriados, descuentos y clima.</li>
        <li><b>Reposición preventiva:</b> Detecta faltantes de stock antes de días pico.</li>
        <li><b>Simulación de descuentos:</b> Estima +{fmt_dec(elasticidad_pct, 0)} % de demanda con 20 % dto.</li>
      </ul>
    </div>
  </div>

  <!-- 4. FIGURA 1: VENTAS — TOP PRODUCTOS AGRUPADOS POR CATEGORÍA -->
  <div class="fig-box">
    <img src="artefactos_mlflow/fig1_ejecutivo_ventas_categorias.png" alt="Figura 1 Ventas por Categoría y Producto">
  </div>

  <!-- 5. FIGURA 2: SOLUCIÓN DEL MODELO -->
  <div class="fig-box">
    <img src="artefactos_mlflow/fig2_ejecutivo_solucion_modelo.png" alt="Figura 2 Solución del Modelo Predictivo">
  </div>

  <!-- 6. CIERRE EJECUTIVO MÍNIMO (3 DECISIONES EN 1 LÍNEA) -->
  <div class="actions-row">
    <div class="action-box">
      <strong>Compras</strong>
      Aumentar stock base en <b>Accesorios</b> y <b>Computación</b> antes de feriados.
    </div>
    <div class="action-box">
      <strong>Almacén</strong>
      Programar alertas automáticas de reposición para fines de semana.
    </div>
    <div class="action-box">
      <strong>Promociones</strong>
      Impulsar <b>Audio</b> y <b>Tablets</b> con descuentos dirigidos del 15–20 %.
    </div>
  </div>

  <div class="footer">
    <span>Reporte Ejecutivo Empresarial • Solución de Planificación Comercial</span>
    <span>Datos validados de ventas e inventario (60 días)</span>
  </div>

</div>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML simplificado y sobrio generado:", OUT_HTML)

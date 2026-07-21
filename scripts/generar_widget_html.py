"""
Genera un archivo HTML standalone (interactivo, con selectores libres de eje X/Y,
zoom, y panel de series temporales sincronizado) para explorar visualmente la
relación entre CUALQUIER par de señales -- incluida la variable objetivo -- sin
depender de que el notebook/Jupyter soporte JS embebido de forma fiable.

Uso típico desde un notebook:

    from generar_widget_html import generar_widget_exploracion
    generar_widget_exploracion(
        df=df_sim,
        columnas_señal=VARIABLES_XMV + VARIABLES_XMEAS,
        columna_objetivo=VARIABLE_CALIDAD,
        ruta_salida='../artefactos/exploracion_visual.html'
    )

Abre el .html resultante directamente en cualquier navegador -- no necesita
Jupyter, kernel, ni conexión a internet salvo para cargar Chart.js desde CDN.

Diseño: los datos completos se cargan una vez en el navegador; el usuario elige
el par eje X / eje Y que quiere ver en el scatter (instantáneo, sin precalcular
las ~N(N-1)/2 combinaciones posibles), y un panel de series temporales aparte
muestra la forma en el tiempo de las dos señales actualmente seleccionadas --
importante porque el scatter puede esconder relaciones espurias entre dos
señales que simplemente derivan juntas en el tiempo, sin relación causal real.
"""
import json


_PLANTILLA_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Exploración visual de señales</title>
<style>
  :root {{
    --surface-1: #f7f6f3; --surface-2: #ffffff; --border: #e1e0d9; --border-strong: #c3c2b7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781; --radius: 8px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface-1: #1f1f1e; --surface-2: #141413; --border: #2c2c2a; --border-strong: #444441;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--surface-1); color: var(--text-primary);
    margin: 0; padding: 2rem; max-width: 900px; margin-inline: auto;
  }}
  h1 {{ font-size: 18px; font-weight: 500; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 14px; font-weight: 500; color: var(--text-secondary); margin: 0 0 6px; }}
  p.hint {{ font-size: 13px; color: var(--text-muted); margin-top: 0; margin-bottom: 1.5rem; }}
  select {{
    font-family: inherit; font-size: 14px; padding: 8px 12px; border-radius: var(--radius);
    border: 0.5px solid var(--border-strong); background: var(--surface-2); color: var(--text-primary);
  }}
  button {{
    font-family: inherit; font-size: 13px; padding: 8px 14px; border-radius: var(--radius);
    border: 0.5px solid var(--border-strong); background: var(--surface-2); color: var(--text-secondary);
    cursor: pointer;
  }}
  button:hover {{ background: var(--surface-1); }}
  details {{
    background: var(--surface-2); border: 0.5px solid var(--border); border-radius: var(--radius);
    padding: 10px 14px; margin-bottom: 1.5rem; font-size: 13px; color: var(--text-secondary);
  }}
  details summary {{ cursor: pointer; font-weight: 500; color: var(--text-primary); }}
  details ul {{ margin: 8px 0 0; padding-left: 18px; line-height: 1.6; }}
</style>
</head>
<body>
<h1>Comparar cualquier señal contra cualquier otra</h1>
<p class="hint">Arrastra sobre el scatter para hacer zoom (rueda del ratón también). Doble clic para volver a la vista completa.</p>

<details>
<summary>Qué observar en el scatter</summary>
<ul>
  <li><b>Forma de la nube</b>: diagonal = relación lineal; curva = no lineal monótona; U o campana = óptimo intermedio (ningún coeficiente numérico lo detecta bien, solo el ojo).</li>
  <li><b>Dispersión</b>: nube estrecha = relación fuerte y predecible; nube ancha = relación real pero ruidosa.</li>
  <li><b>Outliers</b>: puntos muy alejados del resto pueden ser errores de sensor o eventos reales -- y pueden distorsionar mucho el coeficiente r mostrado.</li>
  <li><b>Clusters</b>: dos o más nubes separadas sugieren una variable oculta (p. ej. dos regímenes de operación distintos) que ninguna tabla de correlación va a mostrarte.</li>
  <li><b>Densidad</b>: zonas con muchos puntos son donde el modelo tiene más datos para aprender; zonas dispersas son casi extrapolación.</li>
</ul>
</details>

<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-bottom: 1.25rem;">
  <div style="display: flex; align-items: center; gap: 8px;">
    <label style="font-size: 13px; color: var(--text-secondary);">Eje X</label>
    <select id="select-x"></select>
  </div>
  <div style="display: flex; align-items: center; gap: 8px;">
    <label style="font-size: 13px; color: var(--text-secondary);">Eje Y</label>
    <select id="select-y"></select>
  </div>
  <button id="btn-swap">Intercambiar ejes</button>
  <button id="btn-reset-zoom">Restablecer zoom</button>
</div>

<h2>Series temporales de las señales seleccionadas</h2>
<div style="position: relative; height: 180px; margin-bottom: 1.5rem; background: var(--surface-2); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 10px;">
  <canvas id="chart-series"></canvas>
</div>
<div id="series-legend" style="display:flex; flex-wrap:wrap; gap:16px; font-size:12px; color:var(--text-secondary); margin: -8px 0 1.5rem;"></div>

<h2 id="diff-label"></h2>
<p class="hint" style="margin-top: -6px;">Cambio paso a paso de la señal del eje X -- útil para variables controladas que oscilan alrededor de un punto de operación: la tendencia global puede salir plana aunque la señal tenga transiciones reales (picos aquí = momentos de cambio, no visibles en la serie de arriba a simple vista).</p>
<div style="position: relative; height: 140px; margin-bottom: 1.5rem; background: var(--surface-2); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 10px;">
  <canvas id="chart-diff"></canvas>
</div>
<div style="display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); margin: -8px 0 1.5rem;">
  <span>Media |diff|: <b id="diff-stat-media" style="color: var(--text-primary);"></b></span>
  <span>Máximo |diff|: <b id="diff-stat-max" style="color: var(--text-primary);"></b></span>
</div>

<h2 id="corr-label"></h2>
<div style="position: relative; height: 420px; background: var(--surface-2); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 12px;">
  <canvas id="chart-main"></canvas>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<script>
const DATA = {data_json};
const ALL_NAMES = {all_names_json};
const TARGET_NAME = "{target_name}";
const SERIES_COLORS = {{}};

let xName = "{default_x}";
let yName = TARGET_NAME;
let chart = null;
let seriesChart = null;
let diffChart = null;

function corr(a, b) {{
  const n = a.length;
  const ma = a.reduce((s,v)=>s+v,0)/n, mb = b.reduce((s,v)=>s+v,0)/n;
  let num=0, da=0, db=0;
  for (let i=0;i<n;i++){{ num += (a[i]-ma)*(b[i]-mb); da += (a[i]-ma)**2; db += (b[i]-mb)**2; }}
  const denom = Math.sqrt(da*db);
  return denom === 0 ? 0 : num / denom;
}}

function normalize(arr) {{
  const min = Math.min(...arr), max = Math.max(...arr);
  const range = max - min || 1;
  return arr.map(v => (v - min) / range);
}}

function populateSelect(sel, current) {{
  sel.innerHTML = "";
  ALL_NAMES.forEach(name => {{
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name === TARGET_NAME ? name + " (objetivo)" : name;
    if (name === current) opt.selected = true;
    sel.appendChild(opt);
  }});
}}

function buildSeriesChart() {{
  if (seriesChart) {{ seriesChart.destroy(); seriesChart = null; }}
  const legend = document.getElementById("series-legend");
  legend.innerHTML = "";
  const names = xName === yName ? [xName] : [xName, yName];
  const colors = ["#2a78d6", "#eb6834"];
  const canvas = document.getElementById("chart-series");

  const datasets = names.map((name, i) => ({{
    label: name, data: normalize(DATA[name]), borderColor: colors[i],
    backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0
  }}));

  seriesChart = new Chart(canvas, {{
    type: "line",
    data: {{ labels: DATA[names[0]].map((_,i)=>i), datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
      scales: {{ x: {{ display: false }}, y: {{ display: false, min: -0.05, max: 1.05 }} }}
    }}
  }});

  names.forEach((name, i) => {{
    const item = document.createElement("span");
    item.style.cssText = "display:flex; align-items:center; gap:4px;";
    item.innerHTML = '<span style="width:10px;height:10px;border-radius:2px;background:' + colors[i] + ';"></span>' + name + " (normalizada 0-1, eje temporal = índice de muestra)";
    legend.appendChild(item);
  }});
}}

function buildDiffChart() {{
  if (diffChart) {{ diffChart.destroy(); diffChart = null; }}
  const canvas = document.getElementById("chart-diff");
  const serie = DATA[xName];
  const diffs = serie.slice(1).map((v,i) => v - serie[i]);
  const mediaAbs = diffs.reduce((s,v)=>s+Math.abs(v),0) / diffs.length;
  const maxAbs = Math.max(...diffs.map(Math.abs));

  document.getElementById("diff-label").textContent = "Derivada de " + xName;
  document.getElementById("diff-stat-media").textContent = mediaAbs.toFixed(4);
  document.getElementById("diff-stat-max").textContent = maxAbs.toFixed(4);

  diffChart = new Chart(canvas, {{
    type: "line",
    data: {{ labels: diffs.map((_,i)=>i), datasets: [{{
      data: diffs, borderColor: "#eb6834", backgroundColor: "rgba(235,104,52,0.08)",
      fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.1
    }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: "rgba(128,128,128,0.15)" }} }} }}
    }}
  }});
}}

function buildChart() {{
  if (chart) {{ chart.destroy(); chart = null; }}
  const canvas = document.getElementById("chart-main");
  const points = DATA[xName].map((v,i) => ({{x:v, y:DATA[yName][i]}}));
  const r = corr(DATA[xName], DATA[yName]).toFixed(3);
  document.getElementById("corr-label").textContent = xName + " vs " + yName + "  (r=" + r + ", n=" + points.length + ")";

  chart = new Chart(canvas, {{
    type: "scatter",
    data: {{ datasets: [{{
      label: xName + " vs " + yName, data: points,
      backgroundColor: "#2a78d655", pointRadius: 3, pointHoverRadius: 5
    }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        zoom: {{
          pan: {{ enabled: true, mode: "xy", modifierKey: "shift" }},
          zoom: {{
            drag: {{ enabled: true, backgroundColor: "rgba(42,120,214,0.15)", borderColor: "#2a78d6" }},
            wheel: {{ enabled: true }},
            pinch: {{ enabled: true }},
            mode: "xy"
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: xName, font: {{size: 12}} }} }},
        y: {{ title: {{ display: true, text: yName, font: {{size: 12}} }} }}
      }}
    }}
  }});

  canvas.ondblclick = () => chart.resetZoom();
}}

function refreshAll() {{
  buildSeriesChart();
  buildDiffChart();
  buildChart();
}}

const selX = document.getElementById("select-x");
const selY = document.getElementById("select-y");
populateSelect(selX, xName);
populateSelect(selY, yName);

selX.addEventListener("change", (e) => {{ xName = e.target.value; refreshAll(); }});
selY.addEventListener("change", (e) => {{ yName = e.target.value; refreshAll(); }});
document.getElementById("btn-swap").addEventListener("click", () => {{
  [xName, yName] = [yName, xName];
  populateSelect(selX, xName);
  populateSelect(selY, yName);
  refreshAll();
}});
document.getElementById("btn-reset-zoom").addEventListener("click", () => {{ if (chart) chart.resetZoom(); }});

refreshAll();
</script>
</body>
</html>
"""


def generar_widget_exploracion(df, columnas_señal, columna_objetivo, ruta_salida,
                                max_señales=30, max_filas=2000):
    """
    Genera un HTML standalone e interactivo: selectores libres de eje X e Y
    (cualquier señal, incluida columna_objetivo, en cualquier eje) con zoom
    por arrastre/rueda, botón de reset, y un panel de series temporales de
    las dos señales actualmente seleccionadas -- para ver si una relación en
    el scatter podría ser espuria (dos señales que solo derivan juntas en el
    tiempo, sin relación causal real).

    max_señales: límite de señales en los desplegables (subir este límite no
                 cuesta más cómputo -- el coste real es max_filas, porque
                 cada señal es un array completo incrustado en el HTML).
    max_filas: si el dataframe tiene más filas, se muestrea (cada punto
               adicional pesa en el tamaño del HTML y en el render del
               navegador -- 2000 puntos por señal es más que suficiente
               para ver la forma de cualquier relación).
    """
    # Evitar duplicados si columna_objetivo también aparece en columnas_señal
    todas = [c for c in columnas_señal if c != columna_objetivo] + [columna_objetivo]
    if len(todas) > max_señales:
        print(f"Aviso: se pasaron {len(todas)} señales (incluyendo el objetivo), recortando a las primeras {max_señales}.")
        todas = todas[:max_señales]
        if columna_objetivo not in todas:
            todas[-1] = columna_objetivo  # aseguramos que el objetivo siempre esté disponible

    df_export = df[todas].dropna()
    if len(df_export) > max_filas:
        df_export = df_export.sample(n=max_filas, random_state=42).sort_index()
        print(f"Aviso: se muestrearon {max_filas} de {len(df)} filas disponibles para mantener el HTML ligero.")

    data_dict = {col: df_export[col].round(4).tolist() for col in todas}
    columnas_señal_finales = [c for c in todas if c != columna_objetivo]
    x_por_defecto = columnas_señal_finales[0] if columnas_señal_finales else columna_objetivo

    html = _PLANTILLA_HTML.format(
        target_name=columna_objetivo,
        data_json=json.dumps(data_dict),
        all_names_json=json.dumps(todas),
        default_x=x_por_defecto,
    )

    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Widget guardado en: {ruta_salida}")
    print(f"Señales disponibles en los selectores: {todas}")
    print(f"Filas: {len(df_export)}")
    print("Ábrelo directamente en un navegador -- no necesita Jupyter ni conexión salvo para cargar Chart.js.")

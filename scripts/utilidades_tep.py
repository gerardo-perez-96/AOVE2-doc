"""
Funciones compartidas entre los notebooks del proyecto TEP.

Se definen aquí (en vez de en cada notebook por separado) las funciones que se usan
en más de un notebook, para no duplicar código -- si hay que corregir un bug o cambiar
el comportamiento, se hace en un solo sitio y todos los notebooks lo heredan.

Uso: desde cualquier notebook en esta carpeta, `from utilidades_tep import *`
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


MAX_LAG_CUBO = 5  # mantener coherente con HORIZONTE_FORECAST en 03_forecasting.ipynb


# --- Correlación cruzada (usadas en 01_diagnostico.ipynb y 02_pipeline_nowcast.ipynb) ---

def correlacion_cruzada_maxima(serie_a, serie_b, max_lag=10):
    """
    Compara serie_a con serie_b desplazada entre -max_lag y +max_lag pasos.
    Devuelve el lag donde la correlación es máxima en valor absoluto, y su valor.
    lag > 0 significa que serie_b va POR DETRÁS de serie_a (reacciona con retardo).
    """
    a = serie_a.values if hasattr(serie_a, 'values') else np.asarray(serie_a)
    b = serie_b.values if hasattr(serie_b, 'values') else np.asarray(serie_b)
    resultados = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xa, xb = a[-lag:], b[:lag]
        elif lag > 0:
            xa, xb = a[:-lag], b[lag:]
        else:
            xa, xb = a, b
        if len(xa) > 1 and np.std(xa) > 0 and np.std(xb) > 0:
            r = np.corrcoef(xa, xb)[0, 1]
            resultados.append((lag, r))
    mejor_lag, mejor_r = max(resultados, key=lambda x: abs(x[1]))
    return mejor_lag, mejor_r


def perfil_correlacion_cruzada(serie_a, serie_b, max_lag=5):
    """
    Igual que correlacion_cruzada_maxima, pero devuelve la correlación en CADA lag
    del rango, no solo la máxima -- para poder inspeccionar la forma completa del
    perfil (¿sube gradualmente hasta un pico, o es un salto brusco en un único lag?).
    """
    a = serie_a.values if hasattr(serie_a, 'values') else np.asarray(serie_a)
    b = serie_b.values if hasattr(serie_b, 'values') else np.asarray(serie_b)
    perfil = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xa, xb = a[-lag:], b[:lag]
        elif lag > 0:
            xa, xb = a[:-lag], b[lag:]
        else:
            xa, xb = a, b
        if len(xa) > 1 and np.std(xa) > 0 and np.std(xb) > 0:
            perfil[lag] = np.corrcoef(xa, xb)[0, 1]
        else:
            perfil[lag] = np.nan
    return perfil


# --- Feature engineering y evaluación base (usadas en 02, 03 y 04) ---

def _lags_potencia(n_lags_max, base=2):
    lags = []
    l = 1
    while l < n_lags_max:
        lags.append(l)
        l *= base
    lags.append(n_lags_max)
    return sorted(set(lags))


def construir_features(df, variables_entrada, usar_lags=True, n_lags=3,
                        espaciado_lags='consecutivo', base_potencia=2,
                        usar_rolling_mean=False, ventana_mean=5,
                        usar_rolling_std=False, ventana_std=5,
                        usar_diff=False):
    """
    Genera columnas de características a partir de variables_entrada.
    Todas las operaciones son causales (rolling/shift solo miran al pasado).

    espaciado_lags: 'consecutivo' -> 1,2,3,...,n_lags
                     'potencia'    -> 1,2,4,8,...,n_lags (termina siempre en n_lags)

    Devuelve: (df_con_features, lista_de_columnas_de_features)
    """
    df_out = df.copy()
    columnas_features = list(variables_entrada)

    if espaciado_lags == 'potencia':
        lista_lags = _lags_potencia(n_lags, base=base_potencia)
    else:
        lista_lags = list(range(1, n_lags + 1))

    # Acumulamos las columnas nuevas en un diccionario y las unimos con UN solo concat al
    # final, en vez de insertarlas una a una -- evita fragmentar el DataFrame (pandas avisa
    # con PerformanceWarning) y es notablemente más rápido cuando esta función se llama
    # muchas veces seguidas, como en construir_features_multi_sim (una vez por simulación).
    nuevas_columnas = {}
    for var in variables_entrada:
        if usar_lags:
            for lag in lista_lags:
                col = f'{var}_lag_{lag}'
                nuevas_columnas[col] = df_out[var].shift(lag)
                columnas_features.append(col)
        if usar_rolling_mean:
            col = f'{var}_roll_mean_{ventana_mean}'
            nuevas_columnas[col] = df_out[var].shift(1).rolling(window=ventana_mean).mean()
            columnas_features.append(col)
        if usar_rolling_std:
            col = f'{var}_roll_std_{ventana_std}'
            nuevas_columnas[col] = df_out[var].shift(1).rolling(window=ventana_std).std()
            columnas_features.append(col)
        if usar_diff:
            col = f'{var}_diff'
            nuevas_columnas[col] = df_out[var].diff()
            columnas_features.append(col)

    df_out = pd.concat([df_out, pd.DataFrame(nuevas_columnas, index=df_out.index)], axis=1)

    return df_out, columnas_features


def evaluar_modelo(X, y, train_frac=0.7, n_estimators=150, max_depth=3, learning_rate=0.1):
    """Split cronológico + entrenamiento XGBoost + RMSE."""
    corte = int(len(X) * train_frac)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    modelo = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {'modelo': modelo, 'rmse': rmse, 'X_test': X_test, 'y_test': y_test,
            'preds': preds, 'corte': corte}


# --- Multi-simulación (usadas en 02_pipeline_nowcast.ipynb y 03_forecasting.ipynb) ---

def construir_features_multi_sim(df_raw, columna_sim, variables_entrada, columna_objetivo=None,
                                   columna_tiempo='sample', **kwargs_features):
    """
    Aplica construir_features() de forma INDEPENDIENTE a cada simulación (evita que los
    lags/rolling/diff crucen la frontera entre simulaciones distintas) y concatena el resultado.

    Si se pasa columna_objetivo, se añade DENTRO del bucle por cada grupo -- así queda
    perfectamente alineada con el resto de columnas sin depender de reconstruir índices
    después de concatenar (que es donde es fácil desalinearse sin que salte ningún error).
    """
    partes = []
    columnas_features = None
    for sim_id, grupo in df_raw.groupby(columna_sim):
        grupo = grupo.sort_values(columna_tiempo).reset_index(drop=True)
        df_feat_sim, cols = construir_features(grupo, variables_entrada, **kwargs_features)
        if columna_objetivo is not None:
            df_feat_sim[columna_objetivo] = grupo[columna_objetivo].values
            # Baseline naive para nowcast: calidad(t-1), calculado DENTRO de cada simulación.
            df_feat_sim[f'{columna_objetivo}_lag1_naive'] = grupo[columna_objetivo].shift(1).values
        df_feat_sim[columna_sim] = sim_id
        partes.append(df_feat_sim)
        columnas_features = cols
    df_concat = pd.concat(partes, ignore_index=True)
    return df_concat, columnas_features


def construir_features_forecast_multi_sim(df_raw, columna_sim, variables_entrada, columna_objetivo,
                                             horizonte, columna_tiempo='sample', **kwargs_features):
    """
    Igual que construir_features_multi_sim, pero además calcula el target desplazado
    (shift(-horizonte)) DENTRO de cada simulación -- evita que el target de las últimas
    filas de una simulación se cuele desde la simulación siguiente.
    """
    partes = []
    columnas_features = None
    nombre_target = f'target_futuro_{columna_objetivo}'
    for sim_id, grupo in df_raw.groupby(columna_sim):
        grupo = grupo.sort_values(columna_tiempo).reset_index(drop=True)
        df_feat_sim, cols = construir_features(grupo, variables_entrada, **kwargs_features)
        df_feat_sim[nombre_target] = grupo[columna_objetivo].shift(-horizonte).values
        df_feat_sim['valor_actual_calidad'] = grupo[columna_objetivo].values
        df_feat_sim[columna_sim] = sim_id
        partes.append(df_feat_sim)
        columnas_features = cols
    df_concat = pd.concat(partes, ignore_index=True)
    return df_concat, columnas_features, nombre_target


def evaluar_modelo_split_por_simulacion(df_feat, cols_feat, columna_objetivo, columna_sim,
                                          simulaciones_train, simulaciones_test,
                                          n_estimators=150, max_depth=3, learning_rate=0.1):
    """
    Igual que evaluar_modelo(), pero el split es por SIMULACIÓN COMPLETA, no cronológico
    sobre una serie concatenada -- el test usa simulaciones que el modelo no vio ni una fila.
    """
    df_train = df_feat[df_feat[columna_sim].isin(simulaciones_train)]
    df_test = df_feat[df_feat[columna_sim].isin(simulaciones_test)]

    X_train, y_train = df_train[cols_feat], df_train[columna_objetivo]
    X_test, y_test = df_test[cols_feat], df_test[columna_objetivo]

    modelo = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {'modelo': modelo, 'rmse': rmse, 'X_test': X_test, 'y_test': y_test, 'preds': preds}


# --- Diagnóstico agregado multi-simulación (usado en 01_diagnostico.ipynb) ---

def correlacion_multi_sim(df_raw, col_a, col_b, columna_sim, simulaciones):
    """
    Calcula la correlación de Pearson entre col_a y col_b POR CADA simulación
    por separado, y devuelve la media y desviación entre simulaciones.

    Una correlación calculada sobre una sola simulación puede reflejar el
    artefacto de esa trayectoria concreta, no una propiedad estable del
    proceso -- std alto entre simulaciones es la señal de alerta de eso
    (media y std comparables en magnitud = la "relación" cambia de signo o
    fuerza según la simulación, no es fiable).

    Devuelve: (media, std, n_simulaciones_validas)
    """
    valores = []
    for sim_id in simulaciones:
        grupo = df_raw[df_raw[columna_sim] == sim_id]
        if grupo[col_a].std() > 0 and grupo[col_b].std() > 0:
            r = np.corrcoef(grupo[col_a], grupo[col_b])[0, 1]
            valores.append(r)
    if not valores:
        return np.nan, np.nan, 0
    return float(np.mean(valores)), float(np.std(valores)), len(valores)


def perfil_correlacion_cruzada_multi_sim(df_raw, col_a, col_b, columna_sim, simulaciones, max_lag=10):
    """
    Igual que perfil_correlacion_cruzada, pero calculado POR CADA simulación
    y agregado (media, std) por cada lag -- mismo principio que
    correlacion_multi_sim, aplicado al perfil completo con desfase.

    Devuelve: dict {lag: (media, std)}
    """
    perfiles = []
    for sim_id in simulaciones:
        grupo = df_raw[df_raw[columna_sim] == sim_id].reset_index(drop=True)
        if grupo[col_a].std() > 0 and grupo[col_b].std() > 0:
            perfiles.append(perfil_correlacion_cruzada(grupo[col_a], grupo[col_b], max_lag=max_lag))
    if not perfiles:
        return {}
    agregado = {}
    for lag in perfiles[0].keys():
        valores_lag = [p[lag] for p in perfiles if not np.isnan(p.get(lag, np.nan))]
        if valores_lag:
            agregado[lag] = (float(np.mean(valores_lag)), float(np.std(valores_lag)))
        else:
            agregado[lag] = (np.nan, np.nan)
    return agregado


def pacf_multi_sim(df_raw, columna, columna_sim, simulaciones, n_lags=25):
    """
    Calcula el PACF POR CADA simulación y devuelve la media y std por lag --
    mismo principio que correlacion_multi_sim, aplicado a la autocorrelación
    parcial de una sola señal.

    Devuelve: (medias, stds) -- ambos arrays de longitud n_lags+1
    """
    from statsmodels.tsa.stattools import pacf as _pacf
    resultados = []
    for sim_id in simulaciones:
        grupo = df_raw[df_raw[columna_sim] == sim_id][columna].values
        if np.std(grupo) > 0 and len(grupo) > n_lags * 2:
            try:
                p = _pacf(grupo, nlags=n_lags, method='ywm')
                resultados.append(p)
            except Exception:
                continue
    if not resultados:
        return np.full(n_lags + 1, np.nan), np.full(n_lags + 1, np.nan)
    matriz = np.array(resultados)
    return matriz.mean(axis=0), matriz.std(axis=0)


def tendencia_multi_sim(df_raw, columna, columna_sim, simulaciones):
    """
    Calcula la fuerza de tendencia (pendiente normalizada, r2) POR CADA
    simulación y agrega media y std -- mismo principio, aplicado a la
    detección de deriva/tendencia de una señal.

    Devuelve: dict con 'cambio_total_vs_rango_media', '..._std', 'r2_media', 'r2_std'
    """
    from scipy import stats as _stats
    cambios, r2s = [], []
    for sim_id in simulaciones:
        serie = df_raw[df_raw[columna_sim] == sim_id][columna].values
        if len(serie) < 3 or np.std(serie) == 0:
            continue
        x = np.arange(len(serie))
        slope, intercept, r_value, p_value, std_err = _stats.linregress(x, serie)
        rango = serie.max() - serie.min()
        pendiente_norm = abs(slope) * len(serie) / rango if rango > 0 else 0
        cambios.append(pendiente_norm)
        r2s.append(r_value ** 2)
    if not cambios:
        return {'cambio_total_vs_rango_media': np.nan, 'cambio_total_vs_rango_std': np.nan,
                'r2_media': np.nan, 'r2_std': np.nan}
    return {
        'cambio_total_vs_rango_media': float(np.mean(cambios)),
        'cambio_total_vs_rango_std': float(np.std(cambios)),
        'r2_media': float(np.mean(r2s)),
        'r2_std': float(np.std(r2s)),
    }


def variabilidad_multi_sim(df_raw, columna, columna_sim, simulaciones):
    """
    Mide la VELOCIDAD DE CAMBIO de una señal (basada en diff), no su tendencia
    lineal global -- calculado por simulación y agregado (media + std).

    Para variables controladas/umbralizadas (típico en un proceso industrial con
    lazos de control activos), la señal oscila alrededor de un punto de operación
    en vez de derivar de forma sostenida: una regresión lineal (tendencia_multi_sim)
    da r2 bajo aunque la señal tenga mucha estructura temporal real, porque no está
    "yendo a ningún sitio" en conjunto -- solo se mueve mucho instante a instante.
    Esta función mide justo eso: cuánto cambia la señal de un paso al siguiente,
    sin asumir ninguna dirección global.

    Devuelve: dict con 'diff_abs_media_media', '..._std', 'diff_abs_max_media', '..._std'
    -- diff_abs_media: velocidad de cambio TÍPICA (promedio de |diff| en la simulación)
    -- diff_abs_max: velocidad de cambio MÁXIMA (el salto más grande en un solo paso)
    """
    diff_medias, diff_maximos = [], []
    for sim_id in simulaciones:
        serie = df_raw[df_raw[columna_sim] == sim_id][columna].values
        if len(serie) < 2:
            continue
        diffs_abs = np.abs(np.diff(serie))
        diff_medias.append(diffs_abs.mean())
        diff_maximos.append(diffs_abs.max())
    if not diff_medias:
        return {'diff_abs_media_media': np.nan, 'diff_abs_media_std': np.nan,
                'diff_abs_max_media': np.nan, 'diff_abs_max_std': np.nan}
    return {
        'diff_abs_media_media': float(np.mean(diff_medias)),
        'diff_abs_media_std': float(np.std(diff_medias)),
        'diff_abs_max_media': float(np.mean(diff_maximos)),
        'diff_abs_max_std': float(np.std(diff_maximos)),
    }


def correlacion_diff_multi_sim(df_raw, col_a, col_b, columna_sim, simulaciones):
    """
    Igual que correlacion_multi_sim, pero sobre la DERIVADA (diff) de ambas
    señales, no sobre sus valores crudos -- mide si dos señales REACCIONAN a la
    vez a los mismos eventos, independientemente de en qué nivel absoluto estén
    o de si comparten una deriva de largo plazo. Correlación cruda alta no implica
    correlación de diffs alta, ni al revés -- son preguntas distintas:

    - correlacion_multi_sim: "¿tienden a estar en niveles parecidos a la vez?"
    - correlacion_diff_multi_sim: "¿cambian a la vez cuando algo pasa?"

    Devuelve: (media, std, n_simulaciones_validas)
    """
    valores = []
    for sim_id in simulaciones:
        grupo = df_raw[df_raw[columna_sim] == sim_id]
        diff_a = np.diff(grupo[col_a].values)
        diff_b = np.diff(grupo[col_b].values)
        if len(diff_a) > 1 and np.std(diff_a) > 0 and np.std(diff_b) > 0:
            r = np.corrcoef(diff_a, diff_b)[0, 1]
            valores.append(r)
    if not valores:
        return np.nan, np.nan, 0
    return float(np.mean(valores)), float(np.std(valores)), len(valores)

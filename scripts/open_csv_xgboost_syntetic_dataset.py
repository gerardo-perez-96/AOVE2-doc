import pandas as pd

def analizar_dataset(ruta_archivo):
    print(f"--- ANALIZANDO: {ruta_archivo} ---\n")
    
    # 1. Cargar el dataset (en un entorno real con archivos gigantes, puedes usar memory_map=True)
    df = pd.read_csv(ruta_archivo)
    
    # 2. Obtener dimensiones básicas
    filas, columnas = df.shape
    
    # 3. Calcular uso de memoria real
    memoria_bytes = df.memory_usage(deep=True).sum()
    memoria_mb = memoria_bytes / (1024 ** 2)
    
    # 4. Mostrar reporte de características
    print(f"📊 Dimensiones del Dataset:")
    print(f"   • Número de filas:    {filas:,}")
    print(f"   • Número de columnas: {columnas}")
    print(f"   • Espacio en memoria: {memoria_mb:.2f} MB")
    print("-" * 40)
    
    # 5. Información sobre tipos de datos de las columnas
    print("📋 Resumen de Tipos de Datos:")
    tipos_conteo = df.dtypes.value_counts()
    for tipo, conteo in tipos_conteo.items():
        print(f"   • {tipo}: {conteo} columnas")
    print("-" * 40)
    
    # 6. Mostrar las primeras 5 columnas y las últimas 5 (para no inundar la pantalla con las 50)
    print("🔍 Vista previa de las columnas del dataset:")
    columnas_lista = list(df.columns)
    if len(columnas_lista) > 10:
        columnas_resumen = columnas_lista[:4] + ["..."] + columnas_lista[-4:]
        print(f"   • Columnas: {', '.join(columnas_resumen)}")
    else:
        print(f"   • Columnas: {', '.join(columnas_lista)}")
    print("-" * 40)
    
    # 7. Vista previa de los datos
    print("💡 Primeras 3 filas de ejemplo:")
    # Mostramos solo el timestamp y los 3 primeros sensores para mantenerlo legible
    columnas_muestra = ['timestamp', 'sensor_1_128bit', 'sensor_2_128bit', 'sensor_3_128bit']
    columnas_disponibles = [c for c in columnas_muestra if c in df.columns]
    
    # Si las columnas generadas en el script anterior no tienen "_128bit", usamos el nombre base
    if not columnas_disponibles:
        columnas_muestra_alt = ['timestamp', 'sensor_1', 'sensor_2', 'sensor_3']
        columnas_disponibles = [c for c in columnas_muestra_alt if c in df.columns]
        
    print(df[columnas_disponibles].head(3).to_string(index=False))
    print("-" * 40)

# Ejemplo de uso:
# Reemplaza con la ruta de tu archivo cuando lo hayas generado completo
analizar_dataset("dataset_xgboost_120_dias.csv")
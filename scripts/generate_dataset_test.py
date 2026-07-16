import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_dataset_120_dias():
    archivo_salida = "dataset_xgboost_120_dias.csv"
    dias = 120
    intervalo_segundos = 5
    num_sensores = 50
    
    # Cálculo exacto de filas: 120 días * 24 h * 60 min * 60 s / 5 s = 2,073,600 filas
    total_filas = int((dias * 24 * 60 * 60) / intervalo_segundos)
    fecha_inicio = datetime(2026, 1, 1, 0, 0, 0)
    
    print(f"Iniciando generación de dataset...")
    print(f"Total de filas a generar: {total_filas}")
    print(f"Sensores: {num_sensores} (Mensajes de 128 bits)")
    
    # Escribimos el dataset por bloques (chunks) para no colapsar la memoria RAM
    chunk_size = 50000
    
    # Crear los nombres de las columnas
    columnas = ['timestamp'] + [f'sensor_{i}_128bit' for i in range(1, num_sensores + 1)]
    
    # Crear el archivo y escribir solo los encabezados
    pd.DataFrame(columns=columnas).to_csv(archivo_salida, index=False)
    
    for i in range(0, total_filas, chunk_size):
        # Determinar cuántas filas quedan por escribir en este bloque
        filas_actuales = min(chunk_size, total_filas - i)
        
        # Generar las marcas de tiempo para el bloque actual
        timestamps = [fecha_inicio + timedelta(seconds=(i + j) * intervalo_segundos) for j in range(filas_actuales)]
        df_chunk = pd.DataFrame({'timestamp': timestamps})
        
        # Generar datos para los 50 sensores
        for s in range(1, num_sensores + 1):
            # Para crear un mensaje de 128 bits, generamos 4 enteros de 32 bits sin signo
            p1 = np.random.randint(0, 4294967295, size=filas_actuales, dtype=np.uint32)
            p2 = np.random.randint(0, 4294967295, size=filas_actuales, dtype=np.uint32)
            p3 = np.random.randint(0, 4294967295, size=filas_actuales, dtype=np.uint32)
            p4 = np.random.randint(0, 4294967295, size=filas_actuales, dtype=np.uint32)
            
            # Formateamos los 4 enteros como una única cadena hexadecimal de 32 caracteres (128 bits)
            # Usamos list comprehension que es lo más eficiente para concatenación de strings en Python
            mensajes_128_bits = [f"{a:08x}{b:08x}{c:08x}{d:08x}" for a, b, c, d in zip(p1, p2, p3, p4)]
            
            df_chunk[f'sensor_{s}_128bit'] = mensajes_128_bits
            
        # Añadir (append) el bloque al archivo CSV
        df_chunk.to_csv(archivo_salida, mode='a', header=False, index=False)
        print(f"Progreso: {i + filas_actuales} / {total_filas} filas completadas...")

    print(f"\n¡Proceso finalizado! El dataset ha sido guardado en: {archivo_salida}")

if __name__ == '__main__':
    generar_dataset_120_dias()
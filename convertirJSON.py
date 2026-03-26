import csv
import json
import os

def procesar_carpeta_csv(ruta_carpeta):
    # Verificamos si la carpeta existe
    if not os.path.exists(ruta_carpeta):
        print(f"La carpeta '{ruta_carpeta}' no existe.")
        return

    # Iteramos sobre cada archivo en la carpeta
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith('.csv'):
            ruta_csv = os.path.join(ruta_carpeta, nombre_archivo)
            # Creamos el nombre del JSON basado en el del CSV
            ruta_json = os.path.join(ruta_carpeta, nombre_archivo.replace('.csv', '.json'))
            
            convertir_archivo(ruta_csv, ruta_json)

def convertir_archivo(csv_path, json_path):
    datos = []
    try:
        with open(csv_path, encoding='utf-8') as csvf:
            lector = csv.DictReader(csvf)
            for fila in lector:
                # Solo tomamos las columnas necesarias
                datos.append({
                    "appid": fila.get("appid"),
                    "title": fila.get("title")
                })
        
        with open(json_path, 'w', encoding='utf-8') as jsonf:
            json.dump(datos, jsonf, indent=4, ensure_ascii=False)
            
        print(f"Convertido: {os.path.basename(csv_path)} -> {os.path.basename(json_path)}")
    
    except Exception as e:
        print(f"Error procesando {csv_path}: {e}")

# --- CONFIGURACIÓN ---
# Pon aquí la ruta de tu carpeta (ejemplo: 'mis_canales' o './datos')
carpeta_objetivo = 'apps_diciembre_2026' 
procesar_carpeta_csv(carpeta_objetivo)
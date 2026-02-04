import json
import time
import os
import logging
from gplay_scraper import GPlayScraper

# --- SILENCIAR LOGS MOLESTOS (OPCIONAL) ---
logging.getLogger("gplay_scraper").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)

# --- CONFIGURACIÓN ---
scraper = GPlayScraper(http_client="curl_cffi")

# Carpetas
CARPETA_ENTRADA = "apps_septiembre 2024"
CARPETA_SALIDA = "apps_scraped_2024"  # <--- Nueva carpeta donde se guardarán los resultados

# Archivos a procesar
ARCHIVOS_JSON = ["mp.audience.2.json", "mp.audience.3.json", "mp.audience.4.json", "mp.audience.5.json"]

# Campos a extraer
CAMPOS_FICHA = ["title", "summary", "description", "genre", "score", "installs", "icon"]

def procesar_inventario():
    # 1. Crear carpeta de salida si no existe
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta creada: {CARPETA_SALIDA}")

    # Verificar carpeta de entrada
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"❌ ERROR: No encuentro la carpeta '{CARPETA_ENTRADA}'.")
        return

    print(f"🚀 Iniciando proceso. Salida en: {CARPETA_SALIDA}/")

    # --- BUCLE PRINCIPAL (ARCHIVO POR ARCHIVO) ---
    for nombre_archivo in ARCHIVOS_JSON:
        ruta_entrada = os.path.join(CARPETA_ENTRADA, nombre_archivo)
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo) # Guardamos con el mismo nombre pero en otra carpeta
        
        print(f"\n📂 Procesando archivo de edad: {nombre_archivo}...")
        
        # Reiniciamos la lista para este archivo específico
        resultados_archivo_actual = [] 
        
        # A. LEER EL JSON
        try:
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                data_cruda = json.load(f)
            
            # Detectar estructura
            if isinstance(data_cruda, dict) and "result" in data_cruda:
                apps_lista = data_cruda["result"]
            elif isinstance(data_cruda, list):
                apps_lista = data_cruda
            else:
                print(f"   ⚠️ Estructura desconocida. Saltando archivo.")
                continue

        except FileNotFoundError:
            print(f"   ⚠️ Archivo no encontrado: {ruta_entrada}")
            continue
        except Exception as e:
            print(f"   ❌ Error leyendo JSON: {e}")
            continue

        # B. PROCESAR APPS DEL ARCHIVO ACTUAL
        count = 0
        total_apps = len(apps_lista)
        
        for item in apps_lista:
            app_id = item.get('appid') or item.get('app_id') or item.get('package_name')
            
            if not app_id:
                continue

            print(f"   [{count+1}/{total_apps}] {app_id} ...", end="")

            try:
                # 1. Metadatos
                try:
                    data_app = scraper.app_get_fields(app_id, CAMPOS_FICHA)
                except Exception:
                    data_app = None # Si falla, asumimos que no existe

                # Validación de seguridad (404)
                if not data_app:
                    print(" 💀 (No encontrada)")
                    continue

                # 2. Reviews
                try:
                    reviews_raw = scraper.get_reviews(app_id, count=5)
                except Exception:
                    reviews_raw = []

                reviews_limpias = []
                if reviews_raw:
                    for r in reviews_raw:
                        reviews_limpias.append({
                            "user": r.get('userName'),
                            "score": r.get('score'),
                            "date": r.get('at'),
                            "text": r.get('content')
                        })

                # 3. Armar objeto
                item_full = {
                    "id": app_id,
                    "titulo_original": item.get('title'),
                    "titulo_store": data_app.get("title"),
                    "genero": data_app.get("genre"),
                    "score": data_app.get("score"),
                    "desc_corta": data_app.get("summary"),
                    "desc_larga": data_app.get("description"),
                    "reviews": reviews_limpias
                }

                resultados_archivo_actual.append(item_full)
                print(" ✅")
                count += 1
                
                # Pausa para evitar ban
                time.sleep(0.2)

            except Exception as e:
                print(f" ⚠️ Error inesperado: {e}")

        # C. GUARDAR ESTE ARCHIVO ESPECÍFICO
        # Esto ocurre AL FINAL de procesar cada archivo de audiencia
        if resultados_archivo_actual:
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados_archivo_actual, f, ensure_ascii=False, indent=4)
            print(f"💾 Guardado: {ruta_salida} ({len(resultados_archivo_actual)} apps)")
        else:
            print(f"⚠️ No se guardaron datos para {nombre_archivo} (quizás todas dieron error).")

    print("\n✨ ¡Proceso completo de todos los archivos!")

if __name__ == "__main__":
    procesar_inventario()
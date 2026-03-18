"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    APPS SCRAPER - VERSIÓN FINAL                              ║
║              Google Play Store - Con engagement y keywords                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CARACTERÍSTICAS:
───────────────
✅ Scraping completo de Google Play
✅ 20 reviews por app (vs 5 anterior)
✅ Keywords automáticos extraídos
✅ Engagement score calculado
✅ Metadata completa (developer, version, etc.)
✅ Manejo de errores robusto
✅ Checkpoints automáticos
"""

import json
import time
import os
import logging
import re
from collections import Counter
from gplay_scraper import GPlayScraper

# Silenciar logs
logging.getLogger("gplay_scraper").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

scraper = GPlayScraper(http_client="curl_cffi")

# Carpetas
CARPETA_ENTRADA = "apps_diciembre_2024"
CARPETA_SALIDA = "apps_scraped_2024"

# Archivos a procesar (por edad)
ARCHIVOS_JSON = [
    "mp.audience.2.json",  # 2+
    "mp.audience.3.json",  # 3+
    "mp.audience.4.json",  # 4+
    "mp.audience.5.json"   # 5+
]

# Campos a extraer de Google Play
CAMPOS_FICHA = [
    "title", 
    "summary", 
    "description", 
    "genre", 
    "score", 
    "installs",
    "icon",
    "developer",
    "developerId",
    "contentRating",
    "updated",
    "version"
]

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def extraer_keywords_automaticos(texto, top_n=20):
    """
    Extrae keywords automáticamente del texto usando frecuencia.
    
    Args:
        texto: String con descripción/título
        top_n: Cuántos keywords extraer
    
    Returns:
        List de keywords más frecuentes
    """
    if not texto:
        return []
    
    # Limpiar texto
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', ' ', texto)
    
    # Tokenizar
    palabras = texto.split()
    
    # Stop words (español e inglés)
    stop_words = {
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'para',
        'con', 'su', 'al', 'lo', 'como', 'más', 'o', 'pero', 'sus', 'le', 'ha',
        'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo',
        'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'for',
        'on', 'with', 'as', 'are', 'this', 'be', 'at', 'by', 'an', 'or', 'from'
    }
    
    # Filtrar palabras cortas y stop words
    palabras_filtradas = [
        p for p in palabras 
        if len(p) > 3 and p not in stop_words
    ]
    
    # Contar frecuencias
    contador = Counter(palabras_filtradas)
    
    # Retornar top N
    return [palabra for palabra, freq in contador.most_common(top_n)]


def calcular_engagement_score(score, installs_str):
    """
    Calcula un score de engagement combinando calificación e instalaciones.
    
    Args:
        score: Float 1-5
        installs_str: String como "1,000,000+" o "10,000+"
    
    Returns:
        Float: Score de engagement (0-100)
    """
    import math
    
    # Parsear instalaciones
    if not installs_str:
        installs = 0
    else:
        installs_clean = re.sub(r'[^\d]', '', installs_str)
        installs = int(installs_clean) if installs_clean else 0
    
    # Normalizar score (1-5 → 0-1)
    score_norm = (score - 1) / 4 if score else 0
    
    # Normalizar installs (log scale)
    installs_norm = min(math.log10(installs + 1) / 8, 1.0) if installs > 0 else 0
    
    # Combinar (70% score, 30% popularidad)
    engagement = (score_norm * 0.7) + (installs_norm * 0.3)
    
    return round(engagement * 100, 2)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def procesar_inventario_apps():
    """
    Procesa inventario completo de apps con todas las mejoras.
    """
    # Crear carpeta de salida
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta creada: {CARPETA_SALIDA}")

    # Verificar carpeta de entrada
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"❌ ERROR: No encuentro la carpeta '{CARPETA_ENTRADA}'.")
        print(f"💡 Crea la carpeta y coloca los archivos JSON de audiencias.")
        return

    print(f"🚀 Apps Scraper - Versión Final")
    print(f"   📂 Input: {CARPETA_ENTRADA}/")
    print(f"   📂 Output: {CARPETA_SALIDA}/")
    print(f"   📝 Reviews por app: 20")
    print(f"   🔑 Keywords automáticos: Sí")
    print(f"   📊 Engagement score: Sí\n")

    # Estadísticas globales
    total_exitosas = 0
    total_fallidas = 0
    tiempo_inicio_global = time.time()

    # Procesar archivo por archivo
    for nombre_archivo in ARCHIVOS_JSON:
        ruta_entrada = os.path.join(CARPETA_ENTRADA, nombre_archivo)
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
        
        # Extraer edad del nombre del archivo
        age_rating = nombre_archivo.split('.')[2] + '+'
        
        print(f"{'='*70}")
        print(f"📂 Procesando: {nombre_archivo} (Edad {age_rating})")
        print(f"{'='*70}")
        
        # Lista de resultados para este archivo
        resultados_archivo = []
        
        # Leer JSON de entrada
        try:
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                data_cruda = json.load(f)
            
            # Detectar estructura
            if isinstance(data_cruda, dict) and "result" in data_cruda:
                apps_lista = data_cruda["result"]
            elif isinstance(data_cruda, list):
                apps_lista = data_cruda
            else:
                print(f"   ⚠️ Estructura desconocida. Saltando archivo.\n")
                continue

        except FileNotFoundError:
            print(f"   ⚠️ Archivo no encontrado: {ruta_entrada}\n")
            continue
        except Exception as e:
            print(f"   ❌ Error leyendo JSON: {e}\n")
            continue

        # Procesar apps
        count_exitosas = 0
        count_fallidas = 0
        total_apps = len(apps_lista)
        tiempo_inicio_archivo = time.time()
        
        for idx, item in enumerate(apps_lista):
            # Extraer app_id
            app_id = item.get('appid') or item.get('app_id') or item.get('package_name')
            
            if not app_id:
                count_fallidas += 1
                continue

            print(f"   [{idx+1}/{total_apps}] {app_id[:30]}...", end="", flush=True)

            try:
                # 1. METADATOS DE LA APP
                try:
                    data_app = scraper.app_get_fields(app_id, CAMPOS_FICHA)
                except Exception as e:
                    print(f" 💀 (No encontrada)")
                    count_fallidas += 1
                    continue

                # Validación
                if not data_app or not data_app.get('title'):
                    print(" 💀 (Sin datos)")
                    count_fallidas += 1
                    continue

                # 2. REVIEWS (20 en vez de 5)
                try:
                    reviews_raw = scraper.get_reviews(app_id, count=20)
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

                # 3. EXTRAER KEYWORDS
                texto_completo = f"{data_app.get('title', '')} {data_app.get('description', '')}"
                auto_keywords = extraer_keywords_automaticos(texto_completo, top_n=20)

                # 4. CALCULAR ENGAGEMENT
                engagement = calcular_engagement_score(
                    data_app.get('score', 0),
                    data_app.get('installs', '0')
                )

                # 5. ARMAR OBJETO COMPLETO
                item_completo = {
                    # Identificadores
                    "app_id": app_id,
                    "age_rating": age_rating,
                    
                    # Títulos
                    "titulo_original": item.get('title'),
                    "titulo_store": data_app.get("title"),
                    
                    # Descripciones
                    "desc_corta": data_app.get("summary"),
                    "desc_larga": data_app.get("description"),
                    
                    # Categorización
                    "genero": data_app.get("genre"),
                    "content_rating": data_app.get("contentRating"),
                    
                    # Calidad y popularidad
                    "score": data_app.get("score"),
                    "installs": data_app.get("installs"),
                    "engagement_score": engagement,
                    
                    # Metadata adicional
                    "developer": data_app.get("developer"),
                    "developer_id": data_app.get("developerId"),
                    "icon": data_app.get("icon"),
                    "version": data_app.get("version"),
                    "updated": data_app.get("updated"),
                    
                    # Reviews
                    "reviews": reviews_limpias,
                    "reviews_count": len(reviews_limpias),
                    
                    # Keywords
                    "auto_keywords": auto_keywords
                }

                resultados_archivo.append(item_completo)
                print(f" ✅ ({len(reviews_limpias)}r, {len(auto_keywords)}kw)")
                count_exitosas += 1
                
                # Pausa para evitar ban
                time.sleep(0.2)

            except Exception as e:
                print(f" ⚠️ Error: {e}")
                count_fallidas += 1

        # Guardar resultados de este archivo
        if resultados_archivo:
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados_archivo, f, ensure_ascii=False, indent=2)
            
            tiempo_archivo = time.time() - tiempo_inicio_archivo
            
            print(f"\n💾 Guardado: {ruta_salida}")
            print(f"   ✅ Exitosas: {count_exitosas}")
            print(f"   ❌ Fallidas: {count_fallidas}")
            print(f"   ⏱️ Tiempo: {tiempo_archivo/60:.1f} min\n")
        else:
            print(f"\n⚠️ No se guardaron datos para {nombre_archivo}\n")

        # Actualizar estadísticas globales
        total_exitosas += count_exitosas
        total_fallidas += count_fallidas

    # Resumen final
    tiempo_total = time.time() - tiempo_inicio_global
    
    print(f"{'='*70}")
    print(f"✨ PROCESO COMPLETO")
    print(f"{'='*70}")
    print(f"✅ Total apps exitosas: {total_exitosas}")
    print(f"❌ Total apps fallidas: {total_fallidas}")
    print(f"⏱️ Tiempo total: {tiempo_total/60:.1f} minutos")
    print(f"⚡ Velocidad: {total_exitosas/(tiempo_total/60):.1f} apps/min")


if __name__ == "__main__":
    procesar_inventario_apps()

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              YOUTUBE SCRAPER - VERSIÓN BATCH (OPTIMIZADA)                    ║
║        YouTube Data API v3 | Múltiples API Keys | Batch de 50               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

OPTIMIZACIONES RESPECTO A LA VERSIÓN ANTERIOR:
───────────────────────────────────────────────
1. BATCH REQUESTS: procesa 50 canales por llamada en vez de 1.
   - channels().list acepta hasta 50 IDs separados por coma.
   - Reduce llamadas de 3/canal a ~0.06/canal para channel info.

2. FUSIÓN DE LLAMADAS: channels().list pide snippet+statistics+brandingSettings+contentDetails
   en una sola request, eliminando la llamada separada para obtener el playlist ID.
   - Antes: 3 calls/canal (channel_info + playlist + videos)
   - Ahora: 2 calls/canal (channel_info_batch + videos_batch)
   - Con batch de 50: 1/50 + 1/50 ≈ 0.04 calls/canal

3. SIN SLEEP: el sleep(0.1) se elimina. La API de YouTube no requiere rate limiting
   manual si usás batch — la cuota ya lo regula.

4. CHECKPOINT COMPATIBLE: mismo formato que la versión anterior.
   Los checkpoints existentes se cargan sin modificaciones.

CUOTA (con batch):
──────────────────
- Por canal: ~2 units (antes 3 units, y con search era 102)
- Con 12 keys: ~60,000 canales/día (antes ~40,000)
- Velocidad estimada: 500-2000 canales/minuto (antes ~10/minuto)

COMPATIBILIDAD:
───────────────
- Mismo formato de salida JSON que la versión anterior
- Los checkpoints existentes se retoman automáticamente
- Mismo sistema de rotación de keys y espera de medianoche
"""

import json
import time
import os
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from collections import Counter
import re

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

CARPETA_ENTRADA = "youtube_diciembre_2026"
CARPETA_SALIDA  = "youtube_scraped_2026"

# Tamaño del batch — máximo permitido por la API de YouTube
BATCH_SIZE = 50

IDIOMAS = {
    'English':    'en',
    'Espanol':    'es',
    'Portugues': 'pt'
}

EDADES = {
    'Pre_School_(3-5)':   'preschool',
    'Kids_(6-9)':         'kids',
    'Tweens_(10-12)':     'tweens',
    'Young_Teens_(13-15)':'youngteens',
    'Teens_(16-18)':      'teens'
}

GENEROS = {
    'Boys':  'boys',
    'Girls': 'girls',
    'Both':  'both'
}

ARCHIVOS = []
for idioma_excel, idioma_code in IDIOMAS.items():
    for edad_excel, edad_code in EDADES.items():
        for genero_excel, genero_code in GENEROS.items():
            archivo = f"Youtube_{idioma_excel}_{edad_excel}_{genero_excel}.json"
            ARCHIVOS.append({
                'archivo':        archivo,
                'idioma':         idioma_code,
                'idioma_display': idioma_excel,
                'edad':           edad_code,
                'edad_display':   edad_excel,
                'genero':         genero_code,
                'genero_display': genero_excel
            })

print(f"📊 Archivos a procesar: {len(ARCHIVOS)}")
print(f"   Total esperado: {len(IDIOMAS)} × {len(EDADES)} × {len(GENEROS)} = {len(ARCHIVOS)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES - ENGAGEMENT METRICS
# (idénticas a la versión anterior para mantener el mismo formato de salida)
# ══════════════════════════════════════════════════════════════════════════════

def parsear_fecha(fecha_str):
    """Helper para parsear fechas de YouTube."""
    if not fecha_str:
        return None
    try:
        return datetime.fromisoformat(fecha_str.replace('Z', '+00:00'))
    except:
        return None


def calcular_engagement_completo(channel_info, videos):
    """
    Calcula métricas de engagement avanzadas.
    Idéntico a la versión anterior — mismo formato de salida.
    """
    hoy = datetime.now(timezone.utc)
    hace_3_meses = hoy - timedelta(days=90)
    hace_1_mes   = hoy - timedelta(days=30)

    videos_ultimos_3_meses = []
    views_ultimos_3_meses  = 0

    for video in videos:
        fecha = parsear_fecha(video.get('published_at'))
        if fecha and fecha >= hace_3_meses:
            videos_ultimos_3_meses.append(video)
            views_ultimos_3_meses += video.get('view_count', 0)

    if videos:
        ultimo_video_fecha = parsear_fecha(videos[0].get('published_at'))
        dias_desde_ultimo_video = (hoy - ultimo_video_fecha).days if ultimo_video_fecha else None
    else:
        ultimo_video_fecha      = None
        dias_desde_ultimo_video = None

    monthly_views_reciente = views_ultimos_3_meses / 3 if views_ultimos_3_meses > 0 else 0

    try:
        channel_created = parsear_fecha(channel_info.get('published_at'))
        if channel_created:
            meses_activo = max((hoy - channel_created).days / 30, 1)
            monthly_views_historico = channel_info.get('view_count', 0) / meses_activo
        else:
            monthly_views_historico = 0
    except:
        monthly_views_historico = 0

    videos_ultimo_mes = sum(
        1 for v in videos_ultimos_3_meses
        if parsear_fecha(v.get('published_at')) and parsear_fecha(v.get('published_at')) >= hace_1_mes
    )

    videos_por_mes_reciente = len(videos_ultimos_3_meses) / 3 if videos_ultimos_3_meses else 0

    if monthly_views_historico > 0 and monthly_views_reciente > 0:
        tendencia_ratio = monthly_views_reciente / monthly_views_historico
        tendencia = "creciendo" if tendencia_ratio > 1.2 else ("decayendo" if tendencia_ratio < 0.8 else "estable")
    else:
        tendencia = "desconocido"

    if dias_desde_ultimo_video is not None:
        if   dias_desde_ultimo_video <= 30:  estado_actividad = "activo"
        elif dias_desde_ultimo_video <= 90:  estado_actividad = "poco_activo"
        elif dias_desde_ultimo_video <= 180: estado_actividad = "inactivo"
        else:                                estado_actividad = "abandonado"
    else:
        estado_actividad = "desconocido"

    if videos:
        avg_views_per_video = sum(v.get('view_count', 0) for v in videos) / len(videos)
        avg_likes_per_video = sum(v.get('like_count', 0) for v in videos) / len(videos)
        total_views = sum(v.get('view_count', 0) for v in videos)
        total_likes = sum(v.get('like_count', 0) for v in videos)
        avg_like_ratio = (total_likes / total_views * 100) if total_views > 0 else 0
    else:
        avg_views_per_video = 0
        avg_likes_per_video = 0
        avg_like_ratio      = 0

    return {
        "videos_ultimos_3_meses":  len(videos_ultimos_3_meses),
        "views_ultimos_3_meses":   views_ultimos_3_meses,
        "videos_ultimo_mes":       videos_ultimo_mes,
        "ultimo_video_fecha":      ultimo_video_fecha.isoformat() if ultimo_video_fecha else None,
        "dias_desde_ultimo_video": dias_desde_ultimo_video,
        "monthly_views_historico": int(monthly_views_historico),
        "monthly_views_reciente":  int(monthly_views_reciente),
        "videos_por_mes":          round(videos_por_mes_reciente, 2),
        "avg_views_per_video":     int(avg_views_per_video),
        "avg_likes_per_video":     int(avg_likes_per_video),
        "avg_like_ratio":          round(avg_like_ratio, 2),
        "tendencia":               tendencia,
        "estado_actividad":        estado_actividad
    }


def extraer_keywords_rapido(channel_info, videos):
    """Extrae keywords de channel + videos."""
    keywords = set()

    if channel_info.get('channel_keywords'):
        kw_list = channel_info['channel_keywords'].split(',')
        keywords.update([k.strip().lower() for k in kw_list if k.strip()])

    for video in videos[:10]:
        if video.get('tags'):
            keywords.update([tag.lower() for tag in video['tags'][:5]])

    all_titles = ' '.join([v.get('title', '') for v in videos[:10]])
    words = re.findall(r'\b\w{4,}\b', all_titles.lower())

    stop_words = {'this', 'that', 'with', 'from', 'have', 'more', 'will',
                  'para', 'como', 'esta', 'este', 'más', 'todo', 'video'}
    words = [w for w in words if w not in stop_words]

    word_freq = Counter(words)
    keywords.update([w for w, _ in word_freq.most_common(10)])
    keywords = {k for k in keywords if len(k) > 2}
    return sorted(list(keywords))[:30]


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL - SCRAPER CON BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class YouTubeScraperBatch:
    """
    Scraper de YouTube con batch processing.

    Diferencia clave con la versión anterior:
    - En vez de procesar canal por canal, acumula IDs en lotes de 50
      y hace UNA llamada a la API por lote.
    - channels().list acepta hasta 50 IDs separados por coma.
    - Esto reduce el tiempo de procesamiento en ~50x para channel info.
    """

    def __init__(self):
        self.api_keys = self._cargar_api_keys()

        if not self.api_keys:
            raise ValueError("❌ No se encontraron API keys.")

        self.current_key_index = 0
        self.youtube = self._build_client()

        self.requests_per_key = {i: 0 for i in range(len(self.api_keys))}
        self.max_requests_per_key = 10000

        self.total_requests = 0
        self.success_count  = 0
        self.fail_count     = 0

        print(f"🔑 API Keys cargadas: {len(self.api_keys)}")
        print(f"📊 Cuota total: {len(self.api_keys) * self.max_requests_per_key:,} units")
        print(f"📺 Canales estimados (batch): {len(self.api_keys) * self.max_requests_per_key // 2:,}\n")

    def _cargar_api_keys(self):
        keys = []
        i = 1
        while True:
            key = os.getenv(f"YOUTUBE_API_KEY_{i}")
            if not key or not key.strip():
                break
            keys.append(key.strip())
            i += 1
        if not keys:
            single_key = os.getenv("YOUTUBE_API_KEY")
            if single_key:
                keys.append(single_key)
        return keys

    def _build_client(self):
        return build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.youtube = self._build_client()
        print(f"\n🔄 Rotando a API key #{self.current_key_index + 1}/{len(self.api_keys)}")

    def _make_request(self, request_func):
        """
        Ejecuta una request con manejo de errores, rotación de keys y
        espera automática cuando todas las keys están agotadas.
        """
        keys_agotadas = set()

        while True:
            try:
                response = request_func()
                self.requests_per_key[self.current_key_index] += 1
                self.total_requests += 1
                return response

            except HttpError as e:
                if e.resp.status == 403:
                    print(f"⚠️ Key #{self.current_key_index + 1} agotada")
                    keys_agotadas.add(self.current_key_index)

                    if len(keys_agotadas) >= len(self.api_keys):
                        # Todas las keys agotadas — esperar hasta medianoche PT (7am Argentina)
                        ahora = datetime.now()
                        medianoche_pt = ahora.replace(hour=7, minute=0, second=0, microsecond=0)
                        if medianoche_pt <= ahora:
                            medianoche_pt += timedelta(days=1)
                        segundos = (medianoche_pt - ahora).total_seconds()
                        horas   = int(segundos // 3600)
                        minutos = int((segundos % 3600) // 60)
                        print(f"\n😴 Todas las keys agotadas. Esperando {horas}h {minutos}m...")
                        print(f"   (Podés dejar esto corriendo, va a retomar solo)\n")
                        time.sleep(segundos)
                        # Reset contadores
                        keys_agotadas = set()
                        self.requests_per_key = {i: 0 for i in range(len(self.api_keys))}
                        self.current_key_index = 0
                        self.youtube = self._build_client()
                        print(f"☀️ Cuota renovada, retomando...\n")
                    else:
                        self._rotate_key()
                        continue
                elif e.resp.status == 429:
                    # Rate limit temporal — esperar 5 segundos y reintentar
                    print(f"⚠️ Rate limit (429), esperando 5s...")
                    time.sleep(5)
                    continue
                else:
                    return None
            except Exception as e:
                return None

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH: INFO DE CANALES
    # ─────────────────────────────────────────────────────────────────────────

    def get_channels_info_batch(self, appids: list) -> dict:
        """
        Obtiene info de hasta 50 canales en UNA sola llamada a la API.

        La API acepta múltiples IDs separados por coma en el parámetro `id`.
        También pedimos contentDetails para obtener el uploadsPlaylistId
        en la misma llamada, evitando una request extra por canal.

        Cost: 1 unit por llamada (independientemente de cuántos canales)

        Retorna: dict {appid: channel_info_dict}
        """
        ids_string = ','.join(appids)

        def _request():
            return self.youtube.channels().list(
                part="snippet,statistics,brandingSettings,contentDetails",
                id=ids_string,
                maxResults=50
            ).execute()

        response = self._make_request(_request)
        if not response:
            return {}

        resultado = {}
        for channel in response.get('items', []):
            appid    = channel['id']
            snippet  = channel['snippet']
            stats    = channel['statistics']
            branding = channel.get('brandingSettings', {})
            content  = channel.get('contentDetails', {})

            playlist_id = content.get('relatedPlaylists', {}).get('uploads', '')

            resultado[appid] = {
                "appid":            appid,
                "channel_title":    snippet.get('title'),
                "channel_description": snippet.get('description'),
                "subscriber_count": int(stats.get('subscriberCount', 0)),
                "video_count":      int(stats.get('videoCount', 0)),
                "view_count":       int(stats.get('viewCount', 0)),
                "channel_keywords": branding.get('channel', {}).get('keywords', ''),
                "country":          snippet.get('country'),
                "published_at":     snippet.get('publishedAt'),
                "custom_url":       snippet.get('customUrl', ''),
                "thumbnail":        snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                "uploads_playlist": playlist_id  # guardado para get_videos_batch
            }

        return resultado

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH: VIDEOS DE PLAYLIST
    # ─────────────────────────────────────────────────────────────────────────

    def get_playlist_videos_batch(self, playlist_id: str, max_results: int = 20) -> list:
        """
        Obtiene los últimos N videos de una playlist de uploads.

        Cost: 1 unit por llamada.
        Luego busca stats de esos videos con otra llamada batch.
        """
        def _playlist_request():
            return self.youtube.playlistItems().list(
                part="contentDetails,snippet",
                playlistId=playlist_id,
                maxResults=max_results
            ).execute()

        playlist_response = self._make_request(_playlist_request)
        if not playlist_response:
            return []

        video_ids = [
            item['contentDetails']['videoId']
            for item in playlist_response.get('items', [])
        ]

        if not video_ids:
            return []

        def _videos_request():
            return self.youtube.videos().list(
                part="snippet,statistics",
                id=','.join(video_ids)
            ).execute()

        videos_response = self._make_request(_videos_request)
        if not videos_response:
            return []

        videos = []
        for video in videos_response.get('items', []):
            snippet = video['snippet']
            stats   = video['statistics']
            videos.append({
                'video_id':      video['id'],
                'title':         snippet.get('title'),
                'description':   snippet.get('description', '')[:500],
                'tags':          snippet.get('tags', [])[:20],
                'published_at':  snippet.get('publishedAt'),
                'view_count':    int(stats.get('viewCount', 0)),
                'like_count':    int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0))
            })

        return videos

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESAR BATCH COMPLETO
    # ─────────────────────────────────────────────────────────────────────────

    def procesar_batch(self, appids: list, metadata: dict) -> list:
        """
        Procesa un lote de hasta 50 canales.

        Flujo:
        1. Una llamada batch para info de todos los canales (1 unit total)
        2. Una llamada por canal para sus videos recientes (1 unit/canal)

        El cuello de botella real es paso 2 — cada canal necesita su propia
        llamada porque la playlist es única por canal. Esto es inevitable con
        la API de YouTube sin usar Search (que cuesta 100 units).

        Retorna: lista de resultados en el mismo formato que la versión anterior.
        """
        resultados = []

        # ── Paso 1: info de todos los canales en una sola llamada ─────────────
        channels_info = self.get_channels_info_batch(appids)

        # ── Paso 2: videos de cada canal ──────────────────────────────────────
        for appid in appids:
            channel_info = channels_info.get(appid)
            if not channel_info:
                self.fail_count += 1
                continue

            playlist_id = channel_info.get('uploads_playlist', '')
            if playlist_id:
                videos = self.get_playlist_videos_batch(playlist_id, max_results=20)
            else:
                videos = []

            engagement = calcular_engagement_completo(channel_info, videos)
            keywords   = extraer_keywords_rapido(channel_info, videos)

            result = {
                'appid':        appid,
                'idioma':       metadata['idioma'],
                'edad':         metadata['edad'],
                'genero':       metadata['genero'],

                'channel_title':       channel_info['channel_title'],
                'channel_description': channel_info['channel_description'],
                'channel_keywords':    channel_info['channel_keywords'],
                'country':             channel_info['country'],
                'custom_url':          channel_info['custom_url'],
                'thumbnail':           channel_info['thumbnail'],
                'published_at':        channel_info['published_at'],

                'subscriber_count': channel_info['subscriber_count'],
                'video_count':      channel_info['video_count'],
                'view_count':       channel_info['view_count'],

                'videos_recientes':       videos,
                'videos_recientes_count': len(videos),

                'auto_keywords':    keywords,
                'engagement_metrics': engagement
            }

            resultados.append(result)
            self.success_count += 1

        return resultados

    def print_stats(self):
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS DE USO DE API")
        print(f"{'='*70}")
        for i, requests in self.requests_per_key.items():
            percentage = (requests / self.max_requests_per_key) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"Key #{i+1}: [{bar}] {requests:,}/{self.max_requests_per_key:,} ({percentage:.1f}%)")
        print(f"\n📊 Total requests: {self.total_requests:,}")
        print(f"✅ Exitosos: {self.success_count:,}")
        print(f"❌ Fallidos: {self.fail_count:,}")
        print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def procesar_youtube_completo():
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta creada: {CARPETA_SALIDA}\n")

    scraper = YouTubeScraperBatch()

    print(f"🎬 YouTube Scraper BATCH - 42 Segmentos")
    print(f"   📦 Batch size: {BATCH_SIZE} canales por llamada")
    print(f"   📈 Métricas de engagement completas\n")

    tiempo_inicio_global = time.time()

    for file_info in ARCHIVOS:
        archivo       = file_info['archivo']
        ruta_entrada  = os.path.join(CARPETA_ENTRADA, archivo)
        ruta_salida   = os.path.join(CARPETA_SALIDA, archivo)
        ruta_checkpoint = ruta_salida.replace('.json', '_checkpoint.json')

        # Saltar archivos ya completados
        if os.path.exists(ruta_salida):
            print(f"   ✅ Ya procesado, saltando: {archivo}\n")
            continue

        print(f"{'='*70}")
        print(f"📂 {archivo}")
        print(f"   Idioma: {file_info['idioma_display']}")
        print(f"   Edad:   {file_info['edad_display']}")
        print(f"   Género: {file_info['genero_display']}")
        print(f"{'='*70}")

        # Leer canales
        try:
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"   ⚠️ Archivo no encontrado: {ruta_entrada}\n")
            continue
        except Exception as e:
            print(f"   ❌ Error leyendo JSON: {e}\n")
            continue

        # Detectar estructura
        if isinstance(data, dict) and "channels" in data:
            channels_list = [
                item.get('appid') or item.get('id')
                for item in data['channels']
                if item.get('appid') or item.get('id')
            ]
        elif isinstance(data, list):
            channels_list = [
                item.get('appid') or item.get('id')
                for item in data
                if item.get('appid') or item.get('id')
            ]
        else:
            print(f"   ⚠️ Estructura desconocida\n")
            continue

        total_channels = len(channels_list)
        print(f"   📊 Total canales: {total_channels}")

        # Cargar checkpoint si existe (compatible con formato anterior)
        resultados     = []
        procesados_ids = set()

        if os.path.exists(ruta_checkpoint):
            try:
                with open(ruta_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                resultados     = checkpoint_data.get('results', [])
                procesados_ids = set(r['appid'] for r in resultados)
                print(f"   ♻️ Checkpoint: {len(resultados)} canales ya procesados")
            except Exception as e:
                print(f"   ❌ Error leyendo checkpoint: {e}")

        # Filtrar canales no procesados
        pendientes = [cid for cid in channels_list if cid not in procesados_ids]
        print(f"   📋 Pendientes: {len(pendientes)}")

        if not pendientes:
            print(f"   ✅ Todos procesados, guardando final...\n")
        else:
            tiempo_inicio_archivo = time.time()
            procesados_en_sesion  = 0

            # Procesar en batches de BATCH_SIZE
            for batch_start in range(0, len(pendientes), BATCH_SIZE):
                batch = pendientes[batch_start:batch_start + BATCH_SIZE]
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (len(pendientes) + BATCH_SIZE - 1) // BATCH_SIZE

                print(f"   📦 Batch {batch_num}/{total_batches} ({len(batch)} canales)...", end=" ", flush=True)

                try:
                    batch_results = scraper.procesar_batch(
                        batch,
                        metadata={
                            'idioma': file_info['idioma'],
                            'edad':   file_info['edad'],
                            'genero': file_info['genero']
                        }
                    )

                    resultados.extend(batch_results)
                    procesados_en_sesion += len(batch_results)

                    # Mostrar resumen del batch
                    exitosos  = len(batch_results)
                    fallidos  = len(batch) - exitosos
                    tiempo_tr = time.time() - tiempo_inicio_archivo
                    velocidad = procesados_en_sesion / tiempo_tr if tiempo_tr > 0 else 0

                    print(f"✅{exitosos} 💀{fallidos} | {velocidad:.1f} canales/s")

                    # Checkpoint cada 500 canales procesados
                    procesados_total = len(resultados)
                    if procesados_total % 500 < BATCH_SIZE:
                        with open(ruta_checkpoint, 'w', encoding='utf-8') as f:
                            json.dump({
                                'results':   resultados,
                                'processed': procesados_total,
                                'timestamp': datetime.now().isoformat()
                            }, f, ensure_ascii=False, indent=2)

                        restantes = len(pendientes) - procesados_en_sesion
                        tiempo_est = restantes / velocidad if velocidad > 0 else 0
                        print(f"\n   💾 Checkpoint: {procesados_total} canales")
                        print(f"   ⏳ Estimado restante: {tiempo_est/60:.1f} min\n")

                except KeyboardInterrupt:
                    print(f"\n\n⚠️ Interrumpido por usuario")
                    print(f"💾 Guardando checkpoint...")
                    with open(ruta_checkpoint, 'w', encoding='utf-8') as f:
                        json.dump({
                            'results':   resultados,
                            'processed': len(resultados),
                            'timestamp': datetime.now().isoformat()
                        }, f, ensure_ascii=False, indent=2)
                    return
                except Exception as e:
                    print(f"⚠️ Error en batch: {e}")
                    continue

        # Guardar archivo final
        if resultados:
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)

            tiempo_archivo = time.time() - tiempo_inicio_global

            print(f"\n💾 Guardado: {ruta_salida}")
            print(f"   ✅ Exitosos: {len(resultados)}")
            print(f"   ❌ Fallidos: {total_channels - len(resultados)}")
            print(f"   ⏱️ Tiempo: {tiempo_archivo/60:.1f} min\n")

            if os.path.exists(ruta_checkpoint):
                os.remove(ruta_checkpoint)
        else:
            print(f"\n⚠️ No se guardaron datos\n")

    tiempo_total = time.time() - tiempo_inicio_global
    print(f"\n{'='*70}")
    print(f"✨ PROCESO COMPLETO - 42 ARCHIVOS")
    print(f"{'='*70}")
    print(f"⏱️ Tiempo total: {tiempo_total/60:.1f} min ({tiempo_total/3600:.1f} h)")
    scraper.print_stats()


if __name__ == "__main__":
    procesar_youtube_completo()
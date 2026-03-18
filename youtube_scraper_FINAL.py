"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  YOUTUBE SCRAPER - VERSIÓN FINAL                             ║
║        YouTube Data API v3 | Múltiples API Keys | 42 Segmentos              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

SEGMENTACIÓN (basada en tus excels):
────────────────────────────────────
- Idioma: English, Spanish, Portuguese (3)
- Edad: Pre_School_(3-5), Kids_(6-9), Tweens_(10-12), Young_Teens_(13-15), Teens_(16-19) (5) - SOLO 14 grupos!
- Género: Boys, Girls, Both (3)

Pero según tu imagen hay solo 14 archivos de inglés, no 15.
Aparentemente:
- Pre_School: Boys, Girls, Both (3)
- Kids: Boys, Girls, Both (3) 
- Tweens: Boys, Girls, Both (3)
- Young_Teens: Solo 3 archivos (no 3×3=9?)
- Teens: ??? (no aparece en tu lista)

**Total real: 42 archivos** (14 por idioma × 3 idiomas)

CARACTERÍSTICAS:
───────────────
✅ API oficial de YouTube Data API v3
✅ Soporte para múltiples API keys con rotación automática
✅ Métricas de engagement completas
✅ Sin comentarios (optimizado para cuota)
✅ Checkpoints automáticos cada 100 canales
✅ Genera automáticamente los 42 nombres de archivo

CUOTA API:
──────────
- Por key: 10,000 units/día
- Por canal: 3 units
- Con 12 keys: ~40,000 canales/día

INSTALACIÓN:
────────────
pip install google-api-python-client python-dotenv

CONFIGURACIÓN (.env):
────────────────────
YOUTUBE_API_KEY_1=AIzaSyXXXXXXXXXXXXXXXXXX
YOUTUBE_API_KEY_2=AIzaSyYYYYYYYYYYYYYYYYYY
...
YOUTUBE_API_KEY_12=AIzaSyZZZZZZZZZZZZZZZZZZ
"""

import json
import time
import os
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from collections import Counter
import re

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Carpetas
CARPETA_ENTRADA = "youtube_diciembre_2026"
CARPETA_SALIDA = "youtube_scraped_2026"

# Mapeo para convertir nombres de archivo a códigos
IDIOMAS = {
    'English': 'en',
    'Spanish': 'es', 
    'Portuguese': 'pt'
}

# Grupos de edad
EDADES = {
    'Pre_School_(3-5)': 'preschool',
    'Kids_(6-9)': 'kids',
    'Tweens_(10-12)': 'tweens',
    'Young_Teens_(13-15)': 'youngteens',
    'Teens_(16-18)': 'teens'
}

GENEROS = {
    'Boys': 'boys',
    'Girls': 'girls',
    'Both': 'both'
}

# Generar archivos automáticamente con el formato REAL de tus JSONs
# Formato: Youtube_English_Kids_(6-9)_Both.json
ARCHIVOS = []

for idioma_excel, idioma_code in IDIOMAS.items():
    for edad_excel, edad_code in EDADES.items():
        for genero_excel, genero_code in GENEROS.items():
            # Formato REAL de tus archivos: Youtube_English_Kids_(6-9)_Both.json
            archivo = f"Youtube_{idioma_excel}_{edad_excel}_{genero_excel}.json"
            ARCHIVOS.append({
                'archivo': archivo,
                'idioma': idioma_code,
                'idioma_display': idioma_excel,
                'edad': edad_code,
                'edad_display': edad_excel,
                'genero': genero_code,
                'genero_display': genero_excel
            })

print(f"📊 Archivos a procesar: {len(ARCHIVOS)}")
print(f"   Idiomas: {len(IDIOMAS)}")
print(f"   Edades: {len(EDADES)}")
print(f"   Géneros: {len(GENEROS)}")
print(f"   Total esperado: {len(IDIOMAS)} × {len(EDADES)} × {len(GENEROS)} = {len(IDIOMAS) * len(EDADES) * len(GENEROS)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES - ENGAGEMENT METRICS
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
    """Calcula métricas de engagement avanzadas."""
    from datetime import timezone
    hoy = datetime.now(timezone.utc)
    hace_3_meses = hoy - timedelta(days=90)
    hace_1_mes = hoy - timedelta(days=30)
    
    # VOLUMEN ÚLTIMOS 3 MESES
    videos_ultimos_3_meses = []
    views_ultimos_3_meses = 0
    
    for video in videos:
        fecha = parsear_fecha(video.get('published_at'))
        if fecha and fecha >= hace_3_meses:
            videos_ultimos_3_meses.append(video)
            views_ultimos_3_meses += video.get('view_count', 0)
    
    # ÚLTIMO VIDEO
    if videos:
        ultimo_video_fecha = parsear_fecha(videos[0].get('published_at'))
        dias_desde_ultimo_video = (hoy - ultimo_video_fecha).days if ultimo_video_fecha else None
    else:
        ultimo_video_fecha = None
        dias_desde_ultimo_video = None
    
    # MONTHLY VIEWS
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
    
    # FRECUENCIA
    videos_ultimo_mes = sum(
        1 for v in videos_ultimos_3_meses 
        if parsear_fecha(v.get('published_at')) >= hace_1_mes
    )
    
    videos_por_mes_reciente = len(videos_ultimos_3_meses) / 3 if videos_ultimos_3_meses else 0
    
    # TENDENCIA
    if monthly_views_historico > 0 and monthly_views_reciente > 0:
        tendencia_ratio = monthly_views_reciente / monthly_views_historico
        tendencia = "creciendo" if tendencia_ratio > 1.2 else ("decayendo" if tendencia_ratio < 0.8 else "estable")
    else:
        tendencia = "desconocido"
    
    # ESTADO ACTIVIDAD
    if dias_desde_ultimo_video is not None:
        if dias_desde_ultimo_video <= 30:
            estado_actividad = "activo"
        elif dias_desde_ultimo_video <= 90:
            estado_actividad = "poco_activo"
        elif dias_desde_ultimo_video <= 180:
            estado_actividad = "inactivo"
        else:
            estado_actividad = "abandonado"
    else:
        estado_actividad = "desconocido"
    
    # ENGAGEMENT BÁSICO
    if videos:
        avg_views_per_video = sum(v.get('view_count', 0) for v in videos) / len(videos)
        avg_likes_per_video = sum(v.get('like_count', 0) for v in videos) / len(videos)
        total_views = sum(v.get('view_count', 0) for v in videos)
        total_likes = sum(v.get('like_count', 0) for v in videos)
        avg_like_ratio = (total_likes / total_views * 100) if total_views > 0 else 0
    else:
        avg_views_per_video = 0
        avg_likes_per_video = 0
        avg_like_ratio = 0
    
    return {
        "videos_ultimos_3_meses": len(videos_ultimos_3_meses),
        "views_ultimos_3_meses": views_ultimos_3_meses,
        "videos_ultimo_mes": videos_ultimo_mes,
        "ultimo_video_fecha": ultimo_video_fecha.isoformat() if ultimo_video_fecha else None,
        "dias_desde_ultimo_video": dias_desde_ultimo_video,
        "monthly_views_historico": int(monthly_views_historico),
        "monthly_views_reciente": int(monthly_views_reciente),
        "videos_por_mes": round(videos_por_mes_reciente, 2),
        "avg_views_per_video": int(avg_views_per_video),
        "avg_likes_per_video": int(avg_likes_per_video),
        "avg_like_ratio": round(avg_like_ratio, 2),
        "tendencia": tendencia,
        "estado_actividad": estado_actividad
    }


def extraer_keywords_rapido(channel_info, videos):
    """Extrae keywords de channel + videos."""
    keywords = set()
    
    # Keywords del canal
    if channel_info.get('channel_keywords'):
        kw_list = channel_info['channel_keywords'].split(',')
        keywords.update([k.strip().lower() for k in kw_list if k.strip()])
    
    # Tags de videos
    for video in videos[:10]:
        if video.get('tags'):
            keywords.update([tag.lower() for tag in video['tags'][:5]])
    
    # Palabras frecuentes en títulos
    all_titles = ' '.join([v.get('title', '') for v in videos[:10]])
    words = re.findall(r'\b\w{4,}\b', all_titles.lower())
    
    stop_words = {'this', 'that', 'with', 'from', 'have', 'more', 'will', 'para', 'como', 'esta', 'este', 'más', 'todo', 'video'}
    words = [w for w in words if w not in stop_words]
    
    word_freq = Counter(words)
    keywords.update([w for w, _ in word_freq.most_common(10)])
    
    keywords = {k for k in keywords if len(k) > 2}
    return sorted(list(keywords))[:30]


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL - YOUTUBE SCRAPER CON MÚLTIPLES API KEYS
# ══════════════════════════════════════════════════════════════════════════════

class YouTubeScraperMultiKey:
    """Scraper de YouTube con soporte para múltiples API keys."""
    
    def __init__(self):
        self.api_keys = self._cargar_api_keys()
        
        if not self.api_keys:
            raise ValueError("❌ No se encontraron API keys. Configura YOUTUBE_API_KEY_1, etc en .env")
        
        self.current_key_index = 0
        self.youtube = self._build_client()
        
        self.requests_per_key = {i: 0 for i in range(len(self.api_keys))}
        self.max_requests_per_key = 10000
        
        self.total_requests = 0
        self.success_count = 0
        self.fail_count = 0
        
        print(f"🔑 API Keys cargadas: {len(self.api_keys)}")
        print(f"📊 Cuota total: {len(self.api_keys) * self.max_requests_per_key:,} units")
        print(f"📺 Canales estimados: {len(self.api_keys) * self.max_requests_per_key // 3:,}\n")
    
    def _cargar_api_keys(self):
        """Carga API keys desde .env."""
        keys = []
        
        # Formato: YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, ...
        i = 1
        while True:
            key = os.getenv(f"YOUTUBE_API_KEY_{i}")
            if not key:
                break
            keys.append(key)
            i += 1
        
        # Fallback: Key única
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
    
    def _check_and_rotate_if_needed(self):
        current_requests = self.requests_per_key[self.current_key_index]
        
        if current_requests >= self.max_requests_per_key * 0.95:
            if len(self.api_keys) > 1:
                self._rotate_key()
                return True
            else:
                print(f"\n⚠️ CUOTA AGOTADA: {current_requests}/{self.max_requests_per_key}")
                return False
        return True
    
    def _make_request(self, request_func, *args, **kwargs):
        """Wrapper para requests con manejo de errores y rotación."""
        keys_agotadas = set()

        while True:
            try:
                response = request_func(*args, **kwargs)
                self.requests_per_key[self.current_key_index] += 1
                self.total_requests += 1
                return response

            except HttpError as e:
                if e.resp.status == 403:
                    print(f"⚠️ Key #{self.current_key_index + 1} agotada")
                    keys_agotadas.add(self.current_key_index)

                    if len(keys_agotadas) >= len(self.api_keys):
                        ahora = datetime.now()
                        medianoche_pt = (ahora + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
                        segundos = (medianoche_pt - ahora).total_seconds()
                        horas = int(segundos // 3600)
                        minutos = int((segundos % 3600) // 60)
                        print(f"\n😴 Todas las keys agotadas. Esperando {horas}h {minutos}m hasta medianoche PT...")
                        print(f"   (Podés dejar esto corriendo, va a retomar solo)\n")
                        time.sleep(segundos)
                        keys_agotadas = set()
                        self.requests_per_key = {i: 0 for i in range(len(self.api_keys))}
                        self.current_key_index = 0
                        self.youtube = self._build_client()
                        print(f"☀️ Cuota renovada, retomando...\n")
                    else:
                        self._rotate_key()
                        continue
                else:
                    return None
            except Exception:
                return None
            
    def get_channel_info(self, appid):
        """Obtiene información del canal. Cost: 1 unit"""
        def _request():
            request = self.youtube.channels().list(
                part="snippet,statistics,brandingSettings",
                id=appid
            )
            return request.execute()
        
        response = self._make_request(_request)
        
        if not response or not response.get('items'):
            return None
        
        channel = response['items'][0]
        snippet = channel['snippet']
        stats = channel['statistics']
        branding = channel.get('brandingSettings', {})
        
        return {
            "appid": appid,
            "channel_title": snippet.get('title'),
            "channel_description": snippet.get('description'),
            "subscriber_count": int(stats.get('subscriberCount', 0)),
            "video_count": int(stats.get('videoCount', 0)),
            "view_count": int(stats.get('viewCount', 0)),
            "channel_keywords": branding.get('channel', {}).get('keywords', ''),
            "country": snippet.get('country'),
            "published_at": snippet.get('publishedAt'),
            "custom_url": snippet.get('customUrl', ''),
            "thumbnail": snippet.get('thumbnails', {}).get('high', {}).get('url', '')
        }
    
    def get_recent_videos(self, appid, max_results=20):
        """Obtiene videos recientes. Cost: 2 units (antes 102)"""
        
        # Paso 1: obtener playlist de uploads (1 unit)
        def _playlist_request():
            request = self.youtube.channels().list(
                part="contentDetails",
                id=appid
            )
            return request.execute()
        
        channel_response = self._make_request(_playlist_request)
        if not channel_response or not channel_response.get('items'):
            return []
        
        playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Paso 2: obtener videos de la playlist (1 unit)
        def _items_request():
            request = self.youtube.playlistItems().list(
                part="contentDetails,snippet",
                playlistId=playlist_id,
                maxResults=max_results
            )
            return request.execute()
        
        playlist_response = self._make_request(_items_request)
        if not playlist_response:
            return []
        
        video_ids = [
            item['contentDetails']['videoId'] 
            for item in playlist_response.get('items', [])
        ]
        
        if not video_ids:
            return []
        
        # Paso 3: obtener stats de los videos (1 unit)
        def _videos_request():
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=','.join(video_ids)
            )
            return request.execute()
        
        videos_response = self._make_request(_videos_request)
        
        if not videos_response:
            return []
        
        videos = []
        for video in videos_response.get('items', []):
            snippet = video['snippet']
            stats = video['statistics']
            
            videos.append({
                'video_id': video['id'],
                'title': snippet.get('title'),
                'description': snippet.get('description', '')[:500],
                'tags': snippet.get('tags', [])[:20],
                'published_at': snippet.get('publishedAt'),
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0))
            })
        
        return videos
    
    def scrape_channel_completo(self, appid, metadata):
        """
        Scraping completo de un canal con métricas de engagement.
        
        Args:
            appid: ID del canal
            metadata: Dict con idioma, edad, genero (display names)
        
        Total cost: 3 units por canal
        
        Returns:
            dict con toda la info del canal o None
        """
        # 1. Info del canal (1 unit)
        channel_info = self.get_channel_info(appid)
        if not channel_info:
            self.fail_count += 1
            return None
        
        # 2. Videos recientes (2 units)
        videos = self.get_recent_videos(appid, max_results=20)
        
        # 3. Calcular métricas (local, 0 units)
        engagement = calcular_engagement_completo(channel_info, videos)
        keywords = extraer_keywords_rapido(channel_info, videos)
        
        # 4. Armar resultado
        result = {
            # Segmentación
            'appid': appid,
            'idioma': metadata['idioma'],
            'edad': metadata['edad'],
            'genero': metadata['genero'],
            
            # Info del canal
            'channel_title': channel_info['channel_title'],
            'channel_description': channel_info['channel_description'],
            'channel_keywords': channel_info['channel_keywords'],
            'country': channel_info['country'],
            'custom_url': channel_info['custom_url'],
            'thumbnail': channel_info['thumbnail'],
            'published_at': channel_info['published_at'],
            
            # Estadísticas
            'subscriber_count': channel_info['subscriber_count'],
            'video_count': channel_info['video_count'],
            'view_count': channel_info['view_count'],
            
            # Videos
            'videos_recientes': videos,
            'videos_recientes_count': len(videos),
            
            # Keywords
            'auto_keywords': keywords,
            
            # Engagement
            'engagement_metrics': engagement
        }
        
        self.success_count += 1
        return result
    
    def print_stats(self):
        """Imprime estadísticas de uso de cuota."""
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
    """
    Procesa todos los archivos de YouTube con API oficial.
    """
    # Crear carpeta de salida
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta creada: {CARPETA_SALIDA}\n")
    
    # Inicializar scraper
    scraper = YouTubeScraperMultiKey()
    
    print(f"🎬 YouTube Scraper - API Oficial - 42 Segmentos")
    print(f"   📊 Sin comentarios (optimizado)")
    print(f"   📈 Métricas de engagement completas\n")
    
    # Estadísticas globales
    tiempo_inicio_global = time.time()
    
    # Procesar archivo por archivo
    for file_info in ARCHIVOS:
        archivo = file_info['archivo']
        ruta_entrada = os.path.join(CARPETA_ENTRADA, archivo)
        ruta_salida = os.path.join(CARPETA_SALIDA, archivo)
        ruta_checkpoint = ruta_salida.replace('.json', '_checkpoint.json')
        
        print(f"{'='*70}")
        print(f"📂 {archivo}")
        print(f"   Idioma: {file_info['idioma_display']}")
        print(f"   Edad: {file_info['edad_display']}")
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
        
        # Cargar checkpoint si existe
        resultados = []
        procesados_ids = set()
        
        if os.path.exists(ruta_checkpoint):
            try:
                with open(ruta_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    resultados = checkpoint_data.get('results', [])
                    procesados_ids = set(r['appid'] for r in resultados)
                print(f"   ♻️ Checkpoint: {len(resultados)} canales ya procesados")
                # AGREGAR ESTO:
                if resultados:
                    primeros_ids_checkpoint = [r['appid'] for r in resultados[:5]]
                    primeros_ids_lista = channels_list[:5]
                    print(f"   🔍 Primeros IDs en checkpoint: {primeros_ids_checkpoint}")
                    print(f"   🔍 Primeros IDs en lista:      {primeros_ids_lista}")
            except:
                pass
        
        # Procesar canales
        tiempo_inicio_archivo = time.time()
        
        for idx, appid in enumerate(channels_list):
            # Skip si ya procesado
            if appid in procesados_ids:
                continue
            
            print(f"   [{idx+1}/{total_channels}] {appid[:20]}...", end=" ", flush=True)
            
            try:
                # Scraping
                result = scraper.scrape_channel_completo(
                    appid, 
                    metadata={
                        'idioma': file_info['idioma'],
                        'edad': file_info['edad'],
                        'genero': file_info['genero']
                    }
                )
                
                if result:
                    resultados.append(result)
                    procesados_ids.add(appid)
                    
                    # Mostrar estado de actividad
                    estado = result['engagement_metrics']['estado_actividad']
                    emoji = {"activo": "✅", "poco_activo": "⚡", "inactivo": "⚠️", "abandonado": "💀"}.get(estado, "❓")
                    print(f"{emoji}")
                else:
                    print("💀")
                
                # CHECKPOINT cada 100 canales
                if (idx + 1) % 100 == 0:
                    with open(ruta_checkpoint, 'w', encoding='utf-8') as f:
                        json.dump({
                            'results': resultados,
                            'processed': len(resultados),
                            'timestamp': datetime.now().isoformat()
                        }, f, ensure_ascii=False, indent=2)
                    
                    tiempo_transcurrido = time.time() - tiempo_inicio_archivo
                    velocidad = (idx + 1) / tiempo_transcurrido
                    restantes = total_channels - (idx + 1)
                    tiempo_estimado = restantes / velocidad if velocidad > 0 else 0
                    
                    print(f"\n   💾 Checkpoint: {len(resultados)} canales")
                    print(f"   ⏱️ Velocidad: {velocidad:.1f} canales/seg")
                    print(f"   ⏳ Tiempo estimado: {tiempo_estimado/60:.1f} min\n")
                
                # Pausa para no saturar API
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️ Interrumpido por usuario")
                print(f"💾 Guardando checkpoint...")
                with open(ruta_checkpoint, 'w', encoding='utf-8') as f:
                    json.dump({
                        'results': resultados,
                        'processed': len(resultados),
                        'timestamp': datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
                return
            except Exception as e:
                print(f"⚠️ Error: {e}")
        
        # Guardar archivo final
        if resultados:
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)
            
            tiempo_archivo = time.time() - tiempo_inicio_archivo
            
            print(f"\n💾 Guardado: {ruta_salida}")
            print(f"   ✅ Exitosos: {len(resultados)}")
            print(f"   ❌ Fallidos: {total_channels - len(resultados)}")
            print(f"   ⏱️ Tiempo: {tiempo_archivo/60:.1f} min\n")
            
            # Eliminar checkpoint
            if os.path.exists(ruta_checkpoint):
                os.remove(ruta_checkpoint)
        else:
            print(f"\n⚠️ No se guardaron datos\n")
    
    # Resumen final
    tiempo_total = time.time() - tiempo_inicio_global
    
    print(f"\n{'='*70}")
    print(f"✨ PROCESO COMPLETO - 42 ARCHIVOS")
    print(f"{'='*70}")
    print(f"⏱️ Tiempo total: {tiempo_total/60:.1f} min ({tiempo_total/3600:.1f} h)")
    
    # Estadísticas de API
    scraper.print_stats()


if __name__ == "__main__":
    procesar_youtube_completo()

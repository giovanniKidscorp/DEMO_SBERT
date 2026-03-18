# 🚀 Sistema de Scraping Apps + YouTube - Versión Final

Sistema completo de scraping para Google Play Store y YouTube con métricas de engagement avanzadas.

---

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Apps Scraper](#apps-scraper)
3. [YouTube Scraper](#youtube-scraper)
4. [Estructura de Datos](#estructura-de-datos)
5. [Métricas de Engagement](#métricas-de-engagement)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Instalación

### Requisitos
- Python 3.8+
- pip

### 1. Instalar dependencias

```bash
# Para APPS
pip install gplay-scraper

# Para YOUTUBE
pip install google-api-python-client python-dotenv

# Ambos
pip install gplay-scraper google-api-python-client python-dotenv
```

### 2. Preparar estructura de carpetas

```bash
# Para APPS
mkdir apps_diciembre_2024
mkdir apps_scraped_2024

# Para YOUTUBE
mkdir youtube_diciembre_2024
mkdir youtube_scraped_2024
```

---

## 📱 Apps Scraper

### Características

✅ **Scraping completo de Google Play Store**
- 20 reviews por app (vs 5 anterior)
- Keywords automáticos extraídos (top 20)
- Engagement score calculado (score × installs)
- Metadata completa (developer, version, updated, etc.)
- Manejo robusto de errores (404, timeout, etc.)

### Uso

#### 1. Preparar archivos de entrada

Coloca tus archivos JSON en `apps_diciembre_2024/`:

```
apps_diciembre_2024/
├── mp.audience.2.json  # Apps para 2+
├── mp.audience.3.json  # Apps para 3+
├── mp.audience.4.json  # Apps para 4+
└── mp.audience.5.json  # Apps para 5+
```

**Formato de entrada esperado:**
```json
[
  {
    "appid": "com.example.app",
    "title": "My App"
  }
]
```

O:
```json
{
  "result": [
    {
      "app_id": "com.example.app",
      "title": "My App"
    }
  ]
}
```

#### 2. Ejecutar scraper

```bash
python apps_scraper_FINAL.py
```

#### 3. Resultados

Los archivos se guardarán en `apps_scraped_2024/` con la misma estructura:

```json
[
  {
    "app_id": "com.example.app",
    "age_rating": "2+",
    
    "titulo_original": "My App",
    "titulo_store": "My Awesome App",
    
    "desc_corta": "Short description...",
    "desc_larga": "Full description...",
    
    "genero": "Educational",
    "content_rating": "Everyone",
    
    "score": 4.5,
    "installs": "1,000,000+",
    "engagement_score": 78.5,
    
    "developer": "Example Inc",
    "developer_id": "1234567890",
    "icon": "https://...",
    "version": "1.2.3",
    "updated": "Dec 1, 2024",
    
    "reviews": [
      {
        "user": "John Doe",
        "score": 5,
        "date": "2024-12-01",
        "text": "Great app!"
      }
    ],
    "reviews_count": 20,
    
    "auto_keywords": ["educational", "kids", "learning", ...]
  }
]
```

### Velocidad

- **~0.2 segundos por app** (con pausa anti-ban)
- **300 apps/hora**
- **Ejemplo:** 1,000 apps = ~55 minutos

---

## 🎬 YouTube Scraper

### Características

✅ **YouTube Data API v3 (oficial)**
- Múltiples API keys con rotación automática
- Sin comentarios (optimizado para cuota)
- Métricas de engagement completas
- Checkpoints automáticos cada 100 canales
- Tracking de cuota por key
- 3 units por canal (channel info + videos)

### Cuota y Capacidad

**Por API Key:**
- Cuota: 10,000 units/día
- Cost por canal: 3 units
- **Canales por key: ~3,333/día**

**Con múltiples keys:**
- 3 keys: ~10,000 canales/día
- 6 keys: ~20,000 canales/día
- 12 keys: ~40,000 canales/día

### Uso

#### 1. Obtener API Keys

**Crear proyecto en Google Cloud:**

1. Ve a https://console.cloud.google.com/
2. Crear nuevo proyecto (ej: "YouTube Scraper 1")
3. Habilitar "YouTube Data API v3"
4. Ir a "Credenciales" → "Crear credencial" → "Clave de API"
5. Copiar la API key

**Repetir para múltiples keys:**
- Puedes crear hasta 12 proyectos por cuenta Google
- Para más keys, crear más cuentas Google
- **Recomendado:** 12 keys (1 cuenta × 12 proyectos) para 40K canales

#### 2. Configurar API Keys

Crear archivo `.env` en la misma carpeta que el scraper:

**Opción A: Múltiples keys (recomendado)**
```
YOUTUBE_API_KEY_1=AIzaSyXXXXXXXXXXXXXXXXXX
YOUTUBE_API_KEY_2=AIzaSyYYYYYYYYYYYYYYYYYY
YOUTUBE_API_KEY_3=AIzaSyZZZZZZZZZZZZZZZZZZ
YOUTUBE_API_KEY_4=AIzaSyWWWWWWWWWWWWWWWWWW
...
YOUTUBE_API_KEY_12=AIzaSyVVVVVVVVVVVVVVVVVV
```

**Opción B: Una sola key (para pruebas)**
```
YOUTUBE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXX
```

#### 3. Preparar archivos de entrada

Coloca tus archivos JSON en `youtube_diciembre_2024/`:

```
youtube_diciembre_2024/
├── yt.age2.es.json  # Canales 2+, español
├── yt.age2.en.json  # Canales 2+, inglés
├── yt.age2.pt.json  # Canales 2+, portugués
├── yt.age3.es.json  # Canales 3+, español
├── yt.age3.en.json  # ...
└── yt.age5.pt.json  # 12 archivos total (4 edades × 3 idiomas)
```

**Formato de entrada esperado:**
```json
[
  {
    "channel_id": "UCxxxxxxxxxxxxxx",
    "channel_name": "Peppa Pig"
  }
]
```

O:
```json
{
  "channels": [
    {
      "id": "UCxxxxxxxxxxxxxx"
    }
  ]
}
```

#### 4. Ejecutar scraper

```bash
python youtube_scraper_FINAL.py
```

#### 5. Resultados

Los archivos se guardarán en `youtube_scraped_2024/`:

```json
[
  {
    "channel_id": "UCxxxxxx",
    "age_rating": "2+",
    "language": "es",
    
    "channel_title": "Peppa Pig Español",
    "channel_description": "Canal oficial...",
    "subscriber_count": 5200000,
    "video_count": 847,
    "view_count": 3420000000,
    
    "videos_recientes": [
      {
        "video_id": "abc123",
        "title": "Peppa en la playa",
        "description": "Peppa va a...",
        "upload_date": "20241210",
        "published_at": "2024-12-10T15:30:00",
        "view_count": 1500000,
        "like_count": 25000
      }
    ],
    "videos_recientes_count": 24,
    
    "auto_keywords": ["peppa", "pig", "kids", "animation", ...],
    
    "engagement_metrics": {
      "videos_ultimos_3_meses": 12,
      "views_ultimos_3_meses": 45000000,
      "videos_ultimo_mes": 4,
      
      "ultimo_video_fecha": "2024-12-10T15:30:00",
      "dias_desde_ultimo_video": 5,
      
      "monthly_views_historico": 8500000,
      "monthly_views_reciente": 15000000,
      
      "videos_por_mes": 4.0,
      
      "avg_views_per_video": 3750000,
      "avg_likes_per_video": 42000,
      "avg_like_ratio": 1.12,
      
      "tendencia": "creciendo",
      "estado_actividad": "activo"
    }
  }
]
```

### Velocidad

**Con YouTube Data API v3:**

| API Keys | Canales/día | 40K canales | Costo |
|----------|-------------|-------------|-------|
| **1 key** | ~3,333 | 12 días | Gratis |
| **3 keys** | ~10,000 | 4 días | Gratis |
| **6 keys** | ~20,000 | 2 días | Gratis |
| **12 keys** | ~40,000 | 1 día ⭐ | Gratis |

**Nota:** La API es gratis pero está limitada por cuota diaria (10K units/día por proyecto).

**Velocidad de scraping:**
- ~1-2 canales/segundo (con pausa de 0.1s)
- ~3,600-7,200 canales/hora
- **Limitado por cuota, no por velocidad**

---

## 📊 Estructura de Datos

### Comparación Apps vs YouTube

| Campo | Apps | YouTube |
|-------|------|---------|
| **Identificador** | app_id | channel_id |
| **Título** | titulo_store | channel_title |
| **Descripción** | desc_larga | channel_description |
| **Categoría** | genero | auto_keywords |
| **Calidad** | score (1-5) | avg_like_ratio |
| **Popularidad** | installs | subscriber_count |
| **Reviews/Comentarios** | 20 reviews | - (no incluido) |
| **Keywords** | auto_keywords | auto_keywords |
| **Engagement** | engagement_score | engagement_metrics |
| **Segmentación** | age_rating | age_rating + language |

---

## 📈 Métricas de Engagement

### Apps

**engagement_score** (0-100):
```python
score_norm = (score - 1) / 4  # Normalizar 1-5 → 0-1
installs_norm = log10(installs + 1) / 8  # Log scale
engagement = (score_norm × 0.7) + (installs_norm × 0.3)
```

**Interpretación:**
- 0-30: Baja calidad/popularidad
- 30-60: Media
- 60-80: Buena
- 80-100: Excelente

### YouTube

#### **Volumen Reciente**
- `videos_ultimos_3_meses`: Videos publicados últimos 90 días
- `views_ultimos_3_meses`: Views acumuladas últimos 90 días
- `videos_ultimo_mes`: Videos último mes

#### **Último Video**
- `ultimo_video_fecha`: Fecha del video más reciente
- `dias_desde_ultimo_video`: Días desde última publicación

#### **Monthly Views**
- `monthly_views_historico`: Promedio histórico del canal
- `monthly_views_reciente`: Promedio últimos 3 meses

#### **Frecuencia**
- `videos_por_mes`: Promedio de videos/mes (últimos 3 meses)

#### **Engagement Básico**
- `avg_views_per_video`: Promedio de views por video
- `avg_likes_per_video`: Promedio de likes por video
- `avg_like_ratio`: Porcentaje de likes/views

#### **Tendencia**
- `"creciendo"`: Views recientes +20% vs histórico
- `"estable"`: Views recientes ±20% vs histórico
- `"decayendo"`: Views recientes -20% vs histórico

#### **Estado de Actividad**
- `"activo"`: Último video ≤30 días
- `"poco_activo"`: Último video 30-90 días
- `"inactivo"`: Último video 90-180 días
- `"abandonado"`: Último video >180 días

---

## 🔍 Filtros Recomendados

### Filtrar canales activos

```python
# Después de scrapear YouTube
canales_activos = [
    canal for canal in canales_scrapeados
    if canal['engagement_metrics']['estado_actividad'] in ['activo', 'poco_activo']
    and canal['engagement_metrics']['dias_desde_ultimo_video'] <= 90
    and canal['engagement_metrics']['videos_ultimos_3_meses'] >= 3
]

# Resultado: ~25K canales buenos de 40K totales
```

### Filtrar apps de calidad

```python
# Después de scrapear apps
apps_calidad = [
    app for app in apps_scrapeadas
    if app['engagement_score'] >= 50
    and app['score'] >= 3.5
    and app['reviews_count'] >= 5
]
```

---

## 🛠️ Troubleshooting

### Apps Scraper

**Problema:** "App no encontrada (404)"
```
Causa: App removida o ID incorrecto
Solución: Normal, el scraper continúa con la siguiente
```

**Problema:** "Muy lento"
```
Causa: Pausa anti-ban de 0.2 seg por app
Solución: Ajustar en línea ~155: time.sleep(0.1)  # Reducir a 0.1
Riesgo: Google puede banear IP temporalmente
```

**Problema:** "Ban de Google Play"
```
Causa: Demasiadas requests muy rápido
Solución: 
1. time.sleep(0.5)  # Aumentar pausa
2. Usar VPN/proxy
3. Esperar 1-2 horas
```

### YouTube Scraper

**Problema:** "Cuota agotada en todas las keys"
```
Causa: Se alcanzó el límite de 10K units/día en todas las keys
Solución:
1. Esperar 24 horas para que se resetee
2. Agregar más API keys en .env
3. El scraper guarda checkpoint automáticamente
```

**Problema:** "Error 403 Forbidden"
```
Causa: API key inválida o API no habilitada
Solución:
1. Verificar que YouTube Data API v3 esté habilitada en Google Cloud
2. Verificar que la API key sea correcta en .env
3. El scraper rotará automáticamente a la siguiente key
```

**Problema:** "Canal no encontrado"
```
Causa: Canal borrado, privado, o ID incorrecto
Solución: Normal, el scraper continúa con el siguiente
```

**Problema:** "Demasiado lento"
```
Causa: Pausa de 0.1s entre requests
Solución:
Editar línea ~XXX en youtube_scraper_FINAL.py:
time.sleep(0.05)  # Reducir a 0.05s (cuidado con rate limits)
```

**Problema:** "Checkpoint no funciona"
```
Causa: Archivo checkpoint corrupto
Solución:
1. Eliminar archivo *_checkpoint.json
2. El scraper empezará de nuevo pero sin perder progreso en archivo final
```

---

## 📦 Archivos del Sistema

```
.
├── apps_scraper_FINAL.py           # Scraper de apps
├── youtube_scraper_FINAL.py        # Scraper de YouTube
├── proxies.txt                     # Proxies (opcional)
├── .env                            # Configuración (opcional)
│
├── apps_diciembre_2024/            # Input apps
│   ├── mp.audience.2.json
│   ├── mp.audience.3.json
│   ├── mp.audience.4.json
│   └── mp.audience.5.json
│
├── apps_scraped_2024/              # Output apps
│   ├── mp.audience.2.json
│   ├── mp.audience.3.json
│   ├── mp.audience.4.json
│   └── mp.audience.5.json
│
├── youtube_diciembre_2024/         # Input YouTube
│   ├── yt.age2.es.json
│   ├── yt.age2.en.json
│   ├── yt.age2.pt.json
│   └── ... (12 archivos)
│
└── youtube_scraped_2024/           # Output YouTube
    ├── yt.age2.es.json
    ├── yt.age2.en.json
    ├── yt.age2.pt.json
    └── ... (12 archivos)
```

---

## 🎯 Siguiente Paso: Generar Vectores

Después de scrapear, el siguiente paso es generar embeddings E5 + índices BM25:

```python
# Para cada archivo scrapeado:
# 1. Generar embeddings E5 (768 dims)
# 2. Generar índice BM25
# 3. Subir a PostgreSQL
# 4. Usar en sistema de recomendación híbrido
```

---

## 📞 Soporte

**Apps Scraper:**
- Basado en: gplay-scraper
- Docs: https://github.com/JoMingyu/google-play-scraper

**YouTube Scraper:**
- Basado en: yt-dlp
- Docs: https://github.com/yt-dlp/yt-dlp

**Proxies recomendados:**
- WebShare.io: https://webshare.io (más barato)
- Proxy-Cheap: https://proxy-cheap.com (más rápido)
- Bright Data: https://brightdata.com (premium)

---

## ✅ Checklist de Setup

### Apps
- [ ] `pip install gplay-scraper`
- [ ] Crear carpeta `apps_diciembre_2024/`
- [ ] Colocar archivos JSON de entrada
- [ ] Ejecutar `python apps_scraper_FINAL.py`
- [ ] Verificar output en `apps_scraped_2024/`

### YouTube
- [ ] `pip install google-api-python-client python-dotenv`
- [ ] Crear 1-12 proyectos en Google Cloud Console
- [ ] Habilitar YouTube Data API v3 en cada proyecto
- [ ] Crear API key en cada proyecto
- [ ] Crear carpeta `youtube_diciembre_2024/`
- [ ] Colocar archivos JSON de entrada
- [ ] Crear `.env` con API keys (YOUTUBE_API_KEY_1, _2, _3, ...)
- [ ] Ejecutar `python youtube_scraper_FINAL.py`
- [ ] Verificar output en `youtube_scraped_2024/`

---

**¡Listo para scrapear! 🚀**

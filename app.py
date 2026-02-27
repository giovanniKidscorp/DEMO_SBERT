import streamlit as st
import os
import time
from Classes.recommendation import RecommendationEngine

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Kidscorp Discovery AI",
    page_icon="🚀",
    layout="wide"
)

# --- MAPA DE FUENTES (Ahora viven en la Nube) ---
FUENTES_APPS = {
    "mp.audience.2": "Preescolar (3-5 años)",
    "mp.audience.3": "Niños (6-9 años)",
    "mp.audience.4": "Tweens (10-12 años)",
    "mp.audience.5": "Teens (13-18 años)"
}

def formatear_nombre(fuente_id):
    return FUENTES_APPS.get(fuente_id, fuente_id)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .reportview-container {background: #ffffff;}
    div[data-testid="stExpander"] {border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

# --- CARGA DEL MOTOR (CACHEADO) ---
@st.cache_resource
def load_engine():
    # Inicializamos el motor. 
    # Asegúrate de tener tus variables de entorno configuradas en Streamlit Cloud
    return RecommendationEngine()

engine = load_engine()

# ==========================================
# 🔽 SIDEBAR: CONFIGURACIÓN Y FUENTES 🔽
# ==========================================
with st.sidebar:
    st.title("🎛️ Panel de Control")
    
    fuente_seleccionada = st.radio(
        "Fuente de Datos:",
        ["📱 Apps (Play Store)", "📺 YouTube (Canales DB)"],
        index=0
    )
    
    st.divider()

    # --- CASO A: APPS (DESDE POSTGRES) ---
    if fuente_seleccionada == "📱 Apps (Play Store)":
        st.subheader("☁️ 1. Cargar Datos (Nube)")
        
        # Selector basado en los identificadores de la base de datos
        fuente_id = st.selectbox(
            "Selecciona Audiencia:", 
            list(FUENTES_APPS.keys()), 
            format_func=formatear_nombre
        )
        
        if st.button("🔄 Descargar Dataset", type="primary"):
            with st.spinner(f"Descargando {formatear_nombre(fuente_id)} desde Postgres..."):
                engine._load_from_db(fuente_id)
                engine.source_name = fuente_id  # Guardamos la referencia activa
                
            if not engine.df.empty:
                st.success(f"✅ ¡{len(engine.df)} apps cargadas al instante!")
            else:
                st.error("❌ No se encontraron datos. Revisa la base de datos.")
        
        st.subheader("🛡️ 2. Filtros Activos")
        
        # Verificamos si hay datos cargados de apps para leer los géneros
        if not engine.df.empty and 'genero' in engine.df.columns and engine.source_name in FUENTES_APPS:
            lista_generos = sorted([x for x in engine.df['genero'].unique() if x])
            lista_generos.insert(0, "Todos")
            
            genero_ui = st.selectbox("Filtrar por Género:", lista_generos)
            filtro_genero = genero_ui if genero_ui != "Todos" else None
            st.caption(f"📚 {len(lista_generos)-1} categorías detectadas.")
        else:
            st.info("⚠️ Carga un dataset primero para ver los géneros.")
            filtro_genero = None

        # Slider de Score
        filtro_score = st.slider("Calificación Mínima ⭐", 0.0, 5.0, 4.0, 0.5)

    # --- OPCIÓN B: YOUTUBE (DESDE POSTGRES) ---
    else:
        st.subheader("☁️ Conexión a Base de Datos")
        st.info("Conectando a AWS RDS (Kidscorp Producto)")
        
        # Ya no pedimos límite porque bajamos el bloque exacto procesado
        if st.button("🔄 Conectar y Descargar", type="primary"):
            with st.spinner("Descargando base de YouTube desde la nube..."):
                engine._load_from_db("youtube_channels_db")
                engine.source_name = "youtube_channels_db"
            
            if not engine.df.empty:
                st.success(f"✅ DB Conectada. {len(engine.df)} canales listos.")
            else:
                st.error("❌ No se pudieron cargar datos. Revisa la conexión.")

        filtro_score = 0
        filtro_genero = None


# ==========================================
# 🔽 ÁREA PRINCIPAL: BÚSQUEDA 🔽
# ==========================================

if fuente_seleccionada.startswith("📱"):
    st.title("📱 Keyword Search (Cloud)")
else:
    st.title("📺 Buscador de Canales YouTube (Cloud)")

col_search, col_neg = st.columns([3, 1])
with col_search:
    query = st.text_input("🔎 Concepto a buscar:", placeholder="Ej: aprender matemáticas divirtiéndose")
with col_neg:
    neg_query = st.text_input("⛔ Excluir concepto:", placeholder="Ej: violencia, anuncios")

# Opciones avanzadas
top_k = st.slider("Cantidad de resultados", 1, 10000, 5)

# --- LÓGICA DE EJECUCIÓN ---
if query:
    if engine.embeddings is None:
        st.warning("⚠️ El motor está vacío. Por favor haz clic en 'Descargar Dataset' en el panel lateral.")
    else:
        start_time = time.time()
        
        # Preparamos filtros (Solo si estamos en modo Apps)
        filtros_dict = {}
        if fuente_seleccionada.startswith("📱"):
            filtros_dict['score_min'] = filtro_score
            if filtro_genero:
                filtros_dict['genero'] = filtro_genero
        
        # BUSCAR
        resultados = engine.search(
            query=query, 
            negative_query=neg_query, 
            top_k=top_k, 
            filters=filtros_dict
        )
        
        tiempo = time.time() - start_time
        st.caption(f"⏱️ Encontrados {len(resultados)} resultados en {tiempo:.3f} segundos")

        if not resultados:
            st.info("No se encontraron coincidencias con esos filtros.")

        # --- MOSTRAR RESULTADOS ---
        for item in resultados:
            score_ia = item['score']
            meta = item['metadata']
            titulo = item['titulo']
            desc = item['descripcion']
            
            icono = "🔥" if score_ia > 0.55 else "✨"
            
            if fuente_seleccionada.startswith("📱"):
                # --- MODO APPS ---
                es_app = True
                app_id = meta.get('id') or meta.get('app_id')
                url_destino = f"https://play.google.com/store/apps/details?id={app_id}" if app_id else "#"
                texto_link = "📲 Play Store"
                
                etiqueta_score = f"⭐ {meta.get('score', 'N/A')}"
                etiqueta_centro = meta.get('genero', 'Sin género')
                titulo_centro = "Género"
                kws = ""
                
            else:
                # --- MODO YOUTUBE ---
                es_app = False
                custom_url = meta.get('channel_customurl')
                if custom_url:
                    url_destino = custom_url if custom_url.startswith("http") else f"https://www.youtube.com/{custom_url}"
                    texto_link = "📺 Ver Canal"
                else:
                    url_destino = "#"
                    texto_link = "🚫 Sin Link"
                    
                etiqueta_score = "YouTube" 
                kws = meta.get('channel_bs_ch_keywords', '')
                etiqueta_centro = "Video / Canal"
                titulo_centro = "Tipo"

            # 2. RENDERIZADO VISUAL
            with st.expander(f"{icono} {titulo} (Similitud: {score_ia:.3f})", expanded=True):
                c1, c2, c3 = st.columns([1, 1, 3])
                
                with c1:
                    if es_app:
                        st.metric("Score", etiqueta_score)
                    else:
                        st.markdown(f"#### 📺 Canal")
                    
                    if url_destino != "#":
                        st.link_button(texto_link, url_destino)

                with c2:
                    st.metric(titulo_centro, etiqueta_centro)

                with c3:
                    st.markdown(f"**Descripción:** {desc}")
                    if not es_app and kws:
                        st.caption(f"🏷️ **Keywords:** {kws[:150]}...")
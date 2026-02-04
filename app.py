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
    # Asegúrate de tener tu archivo .env en la misma carpeta para las credenciales de DB
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

    # --- CASO A: APPS ---
    if fuente_seleccionada == "📱 Apps (Play Store)":
        st.subheader("📂 1. Cargar Datos")
        CARPETA_DATOS = "apps_scraped_2024"
        
        # Lógica de carga de archivo (igual que antes)
        if os.path.exists(CARPETA_DATOS):
            archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith('.json')]
            if archivos:
                archivo_json = st.selectbox("Selecciona Audiencia:", archivos)
                
                if st.button("🔄 Cargar Dataset", type="primary"):
                    with st.spinner("Procesando..."):
                        ruta = os.path.join(CARPETA_DATOS, archivo_json)
                        engine.load_from_json(ruta)
                    st.success("✅ Datos actualizados")
            else:
                st.warning("Carpeta vacía.")
        
        # --- AQUI ESTA LA MEJORA DE UI ---
        st.subheader("🛡️ 2. Filtros Activos")
        
        # Verificamos si hay datos cargados para leer los géneros
        if not engine.df.empty and 'genero' in engine.df.columns:
            # 1. Extraemos géneros únicos y limpiamos vacíos
            lista_generos = sorted([x for x in engine.df['genero'].unique() if x])
            # Agregamos opción "Todos" al principio
            lista_generos.insert(0, "Todos")
            
            # 2. Widget de Selección Inteligente
            genero_ui = st.selectbox("Filtrar por Género:", lista_generos)
            
            # Lógica para pasar al motor
            # Si elige "Todos", pasamos None (sin filtro)
            filtro_genero = genero_ui if genero_ui != "Todos" else None
            
            # Métrica visual
            st.caption(f"📚 {len(lista_generos)-1} categorías detectadas en este archivo.")
            
        else:
            st.info("⚠️ Carga un dataset primero para ver los géneros.")
            filtro_genero = None

        # Slider de Score (Siempre visible)
        filtro_score = st.slider("Calificación Mínima ⭐", 0.0, 5.0, 4.0, 0.5)

    # --- OPCIÓN B: YOUTUBE (POSTGRESQL) ---
    else:
        st.subheader("☁️ Conexión a Base de Datos")
        st.info("Conectando a AWS RDS (Kidscorp Youtube)")
        
        # Límite para no traerse millones de canales de golpe
        limit_db = st.number_input("Límite de Canales a analizar", 1000, 50000, 10000, step=1000)
        
        if st.button("🔄 Conectar y Descargar", type="primary"):
            with st.spinner("Conectando a Postgres y generando vectores..."):
                # La función load_from_postgres usa os.getenv para las credenciales
                engine.load_from_postgres(limit=limit_db)
            
            if not engine.df.empty:
                st.success(f"✅ DB Conectada. {len(engine.df)} canales listos.")
            else:
                st.error("❌ No se pudieron cargar datos. Revisa la conexión.")

        # Filtros específicos de YOUTUBE (Si quisieras agregar alguno)
        # Por ahora YouTube no tiene Score o Genero estandarizado en tu tabla
        filtro_score = 0
        filtro_genero = None


# ==========================================
# 🔽 ÁREA PRINCIPAL: BÚSQUEDA 🔽
# ==========================================

# Título dinámico
if fuente_seleccionada.startswith("📱"):
    st.title("📱 Keyword search")
else:
    st.title("📺 Buscador de Canales YouTube")

col_search, col_neg = st.columns([3, 1])
with col_search:
    query = st.text_input("🔎 Concepto a buscar:", placeholder="Ej: aprender matemáticas divirtiéndose")
with col_neg:
    neg_query = st.text_input("⛔ Excluir concepto:", placeholder="Ej: violencia, anuncios")

# Opciones avanzadas
top_k = st.slider("Cantidad de resultados", 1, 1000, 5)

# --- LÓGICA DE EJECUCIÓN ---
if query:
    if engine.embeddings is None:
        st.warning("⚠️ El motor está vacío. Por favor CARGA un dataset en la barra lateral izquierda.")
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

        # --- MOSTRAR RESULTADOS (Adaptativo) ---
        for item in resultados:
            score_ia = item['score']
            meta = item['metadata']
            titulo = item['titulo']
            desc = item['descripcion']
            
            # Icono según score
            icono = "🔥" if score_ia > 0.55 else "✨"
            
            # --- DISEÑO PARA APPS ---
            if fuente_seleccionada.startswith("📱"):
                with st.expander(f"{icono} {titulo} (Similitud: {score_ia})", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 4])
                    with c1: st.metric("Score Store", f"⭐ {meta.get('score', 0)}")
                    with c2: st.metric("Género", meta.get('genero', 'N/A'))
                    with c3: st.markdown(f"**Descripción:** {desc}")
            
            # --- DISEÑO PARA YOUTUBE ---
            else:
                with st.expander(f"📺 {titulo} (Similitud: {score_ia})", expanded=True):
                    # YouTube no tiene score numérico ni género en tu tabla simple
                    st.markdown(f"**Descripción del Canal:** {desc}")
                    
                    # Mostramos keywords si existen
                    kws = meta.get('channel_bs_ch_keywords', '')
                    if kws:
                        st.caption(f"🏷️ **Keywords:** {kws[:100]}...")
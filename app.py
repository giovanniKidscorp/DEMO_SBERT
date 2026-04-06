"""
app.py — Kidscorp Discovery AI
UI unificada para Apps (Play Store) y YouTube.
- Apps: carga desde Postgres o PKL local
- YouTube: carga desde PKL local
"""

import streamlit as st
import os
import glob
import time
from Classes.recommendation import RecommendationEngine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kidscorp Discovery AI",
    page_icon="🚀",
    layout="wide"
)

# Carpeta donde están los PKLs locales
CARPETA_PKL = r"basesTemporales\vectores\content\vectores"

# Fuentes de apps en Postgres
FUENTES_APPS_DB = {
    "mp.audience.2": "Preescolar (3-5 años)",
    "mp.audience.3": "Niños (6-9 años)",
    "mp.audience.4": "Tweens (10-12 años)",
    "mp.audience.5": "Teens (13-18 años)"
}

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border-radius: 8px;
        padding: 8px 12px;
    }
    .breakdown-badge {
        display: inline-block;
        background: #eef2ff;
        color: #3b4bdb;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 4px;
        font-family: monospace;
    }
    .badge-metric {background: #f0fdf4; color: #16a34a;}
    .badge-lexico {background: #fff7ed; color: #c2410c;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def descubrir_segmentos_pkl() -> dict:
    """
    Escanea CARPETA_PKL y retorna segmentos que tienen ambos PKLs.
    Separa YouTube de Apps por el prefijo del nombre.
    Retorna: {nombre_display: nombre_segmento}
    """
    emb_files = glob.glob(os.path.join(CARPETA_PKL, "*_embeddings.pkl"))
    segmentos = {}
    for emb_path in sorted(emb_files):
        nombre    = os.path.basename(emb_path).replace("_embeddings.pkl", "")
        bm25_path = emb_path.replace("_embeddings.pkl", "_bm25.pkl")
        if os.path.exists(bm25_path):
            display = nombre.replace("Youtube_", "📺 ").replace("Apps_", "📱 ").replace("_", " ")
            segmentos[display] = nombre
    return segmentos

def formatear_numero(n) -> str:
    try:
        n = int(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(n)
    except:
        return "N/A"

def estado_badge(estado: str) -> str:
    return {
        "activo":      "🟢 Activo",
        "poco_activo": "🟡 Poco activo",
        "inactivo":    "🟠 Inactivo",
        "abandonado":  "🔴 Abandonado",
        "desconocido": "⚪ Desconocido",
    }.get(str(estado).lower(), f"⚪ {estado}")

def tendencia_badge(tendencia: str) -> str:
    return {
        "creciendo":   "📈 Creciendo",
        "estable":     "➡️ Estable",
        "decayendo":   "📉 Decayendo",
        "desconocido": "❓ Desconocido",
    }.get(str(tendencia).lower(), f"❓ {tendencia}")

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return RecommendationEngine()

engine = load_engine()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎛️ Panel de Control")
    st.divider()

    # ── Modo ──────────────────────────────────────────────────────────────────
    modo = st.radio(
        "Fuente de datos:",
        ["📺 YouTube (PKL local)", "📱 Apps (Postgres)", "📱 Apps (PKL local)"],
        index=0
    )
    st.divider()

    # ── MODO YOUTUBE PKL ──────────────────────────────────────────────────────
    if modo == "📺 YouTube (PKL local)":
        st.subheader("📦 Segmento YouTube")

        segmentos = {k: v for k, v in descubrir_segmentos_pkl().items()
                     if "youtube" in v.lower()}

        if not segmentos:
            st.error(f"❌ No hay PKLs de YouTube en `{CARPETA_PKL}`")
        else:
            st.caption(f"{len(segmentos)} segmento(s) disponibles")
            seg_display = st.selectbox("Segmento:", list(segmentos.keys()))
            nombre_seg  = segmentos[seg_display]

            if st.button("🔄 Cargar", type="primary", key="btn_yt"):
                with st.spinner(f"Cargando {seg_display}..."):
                    engine.load_from_pkl(nombre_seg, carpeta=CARPETA_PKL)
                if engine.embeddings is not None:
                    st.success(f"✅ {len(engine.df):,} canales")
                else:
                    st.error("❌ Error al cargar.")

        filtro_score  = 0.0
        filtro_genero = None

    # ── MODO APPS POSTGRES ────────────────────────────────────────────────────
    elif modo == "📱 Apps (Postgres)":
        st.subheader("☁️ Apps desde Postgres")

        fuente_id = st.selectbox(
            "Audiencia:",
            list(FUENTES_APPS_DB.keys()),
            format_func=lambda x: FUENTES_APPS_DB[x]
        )

        if st.button("🔄 Descargar", type="primary", key="btn_apps_db"):
            with st.spinner(f"Descargando {FUENTES_APPS_DB[fuente_id]}..."):
                engine._load_from_db(fuente_id)
                engine.source_name = fuente_id
            if not engine.df.empty:
                st.success(f"✅ {len(engine.df):,} apps cargadas")
            else:
                st.error("❌ No se encontraron datos.")

        st.divider()
        st.subheader("🛡️ Filtros")

        if not engine.df.empty and 'genero' in engine.df.columns and engine.source_name in FUENTES_APPS_DB:
            lista_generos = sorted([x for x in engine.df['genero'].unique() if x])
            lista_generos.insert(0, "Todos")
            genero_ui     = st.selectbox("Género:", lista_generos)
            filtro_genero = genero_ui if genero_ui != "Todos" else None
            st.caption(f"📚 {len(lista_generos)-1} géneros")
        else:
            st.info("Cargá un dataset para ver filtros.")
            filtro_genero = None

        filtro_score = st.slider("⭐ Calificación mínima", 0.0, 5.0, 4.0, 0.5)

    # ── MODO APPS PKL ─────────────────────────────────────────────────────────
    else:
        st.subheader("📦 Segmento Apps (PKL)")

        segmentos = {k: v for k, v in descubrir_segmentos_pkl().items()
                     if "youtube" not in v.lower()}

        if not segmentos:
            # Si no hay PKLs de apps separados, mostrar todos
            segmentos = {k: v for k, v in descubrir_segmentos_pkl().items()
                         if "youtube" not in v.lower()}

        if not segmentos:
            st.error(f"❌ No hay PKLs de Apps en `{CARPETA_PKL}`")
            filtro_score  = 0.0
            filtro_genero = None
        else:
            st.caption(f"{len(segmentos)} segmento(s) disponibles")
            seg_display = st.selectbox("Segmento:", list(segmentos.keys()), key="seg_apps")
            nombre_seg  = segmentos[seg_display]

            if st.button("🔄 Cargar", type="primary", key="btn_apps_pkl"):
                with st.spinner(f"Cargando {seg_display}..."):
                    engine.load_from_pkl(nombre_seg, carpeta=CARPETA_PKL)
                if engine.embeddings is not None:
                    st.success(f"✅ {len(engine.df):,} apps")
                else:
                    st.error("❌ Error al cargar.")

            st.divider()
            st.subheader("🛡️ Filtros")

            if not engine.df.empty and 'genero' in engine.df.columns:
                lista_generos = sorted([x for x in engine.df['genero'].unique() if x])
                lista_generos.insert(0, "Todos")
                genero_ui     = st.selectbox("Género:", lista_generos, key="gen_apps")
                filtro_genero = genero_ui if genero_ui != "Todos" else None
            else:
                filtro_genero = None

            filtro_score = st.slider("⭐ Calificación mínima", 0.0, 5.0, 4.0, 0.5, key="score_apps")

    st.divider()

    # ── Pesos del score ───────────────────────────────────────────────────────
    st.subheader("⚖️ Pesos del Score")
    w_semantic = st.slider("🧠 Semántico (E5)",  0.0, 1.0, 0.50, 0.05)
    w_lexical  = st.slider("🔤 Léxico (BM25)",   0.0, 1.0, 0.20, 0.05)

    # Slider de métricas solo visible en modo YouTube
    if modo == "📺 YouTube (PKL local)":
        w_metric = st.slider("📊 Métricas (YouTube)", 0.0, 1.0, 0.30, 0.05)
        total_w  = w_semantic + w_lexical + w_metric
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ Pesos suman {total_w:.2f}")
    else:
        w_metric = 0.0
        st.caption("📊 Métricas: N/A (solo YouTube)")

    st.divider()

    # ── Info segmento activo ──────────────────────────────────────────────────
    if engine.embeddings is not None and engine.source_name:
        tipo = "📺 YouTube" if engine.es_youtube else "📱 Apps"
        st.subheader("ℹ️ Activo")
        st.info(
            f"**{engine.source_name}**\n\n"
            f"{tipo} · {len(engine.df):,} registros\n\n"
            f"{'✅ Métricas' if engine.metric_stats else '—'}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# ÁREA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
es_youtube_activo = engine.es_youtube if engine.embeddings is not None else (modo == "📺 YouTube (PKL local)")

if es_youtube_activo:
    st.title("📺 Kidscorp Discovery AI — YouTube")
else:
    st.title("📱 Kidscorp Discovery AI — Apps")

col_search, col_neg = st.columns([3, 1])
with col_search:
    query = st.text_input("🔎 Buscar:", placeholder="Ej: juegos educativos matemáticas")
with col_neg:
    neg_query = st.text_input("⛔ Excluir:", placeholder="Ej: violencia")

col_topk, col_opts = st.columns([2, 1])
with col_topk:
    top_k = st.slider("Cantidad de resultados", 1, 100, 10)
with col_opts:
    mostrar_breakdown     = st.checkbox("Desglose de score", value=True)
    mostrar_desc_completa = st.checkbox("Descripción completa", value=False)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA Y RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
if query:
    if engine.embeddings is None:
        st.warning("⚠️ Cargá un segmento desde el panel lateral primero.")
    else:
        # Armar filtros
        filtros_dict = {}
        if not engine.es_youtube:
            if filtro_score > 0:
                filtros_dict['score_min'] = filtro_score
            if filtro_genero:
                filtros_dict['genero'] = filtro_genero

        start_time = time.time()
        resultados = engine.search(
            query=query,
            negative_query=neg_query if neg_query else None,
            top_k=top_k,
            filters=filtros_dict if filtros_dict else None,
            semantic_weight=w_semantic,
            lexical_weight=w_lexical,
            metric_weight=w_metric,
        )
        tiempo = time.time() - start_time

        st.caption(f"⏱️ {len(resultados)} resultados en {tiempo:.3f}s — `{engine.source_name}`")

        if not resultados:
            st.info("Sin resultados. Probá con otros términos o ajustá los filtros.")

        for rank, item in enumerate(resultados, 1):
            score_ia  = item['score']
            meta      = item['metadata']
            titulo    = item['titulo']
            desc      = item['descripcion']
            breakdown = item.get('score_breakdown', {})
            es_yt     = item.get('es_youtube', engine.es_youtube)

            icono = "🔥" if score_ia > 0.65 else "✨" if score_ia > 0.50 else "💡"

            with st.expander(
                f"{icono} #{rank} — {titulo}  |  Score: {score_ia:.3f}",
                expanded=(rank <= 3)
            ):
                # ── YOUTUBE ───────────────────────────────────────────────────
                if es_yt:
                    custom_url = meta.get('custom_url') or meta.get('channel_customurl', '')
                    url_canal  = (custom_url if custom_url.startswith("http")
                                  else f"https://www.youtube.com/{custom_url}") if custom_url else None

                    subs       = formatear_numero(meta.get('subscriber_count', 0))
                    views      = formatear_numero(meta.get('view_count', 0))
                    avg_views  = formatear_numero(meta.get('avg_views_per_video', 0))
                    videos_mes = meta.get('videos_por_mes', 'N/A')
                    like_ratio = meta.get('avg_like_ratio', 'N/A')
                    estado     = estado_badge(meta.get('estado_actividad', 'desconocido'))
                    tendencia  = tendencia_badge(meta.get('tendencia', 'desconocido'))
                    dias_ult   = meta.get('dias_desde_ultimo_video', 'N/A')
                    pais       = meta.get('country', '')

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: st.metric("👥 Suscriptores", subs)
                    with c2: st.metric("👁️ Views totales", views)
                    with c3: st.metric("📹 Avg views/video", avg_views)
                    with c4: st.metric("📅 Videos/mes", f"{videos_mes:.1f}" if isinstance(videos_mes, float) else videos_mes)
                    with c5: st.metric("❤️ Like ratio", f"{like_ratio:.2f}%" if isinstance(like_ratio, float) else like_ratio)

                    c6, c7, c8, c9 = st.columns(4)
                    with c6: st.metric("🟢 Estado", estado)
                    with c7: st.metric("📈 Tendencia", tendencia)
                    with c8: st.metric("📆 Días sin subir", dias_ult)
                    with c9: st.metric("🌍 País", pais or "N/A")

                    st.divider()
                    col_desc, col_link = st.columns([4, 1])
                    with col_desc:
                        texto_desc = meta.get('channel_description', desc) if mostrar_desc_completa else desc
                        st.markdown(f"**Descripción:** {texto_desc}")
                        kw = meta.get('channel_keywords', '')
                        if kw:
                            st.caption(f"🏷️ **Keywords:** {kw[:200]}")
                    with col_link:
                        if url_canal:
                            st.link_button("📺 Ver Canal", url_canal)
                        idioma = meta.get('idioma', '')
                        edad   = meta.get('edad', '')
                        genero = meta.get('genero', '')
                        if idioma:
                            st.caption(f"🌐 `{idioma}` · `{edad}` · `{genero}`")

                # ── APPS ──────────────────────────────────────────────────────
                else:
                    app_id    = meta.get('id') or meta.get('app_id')
                    url_app   = f"https://play.google.com/store/apps/details?id={app_id}" if app_id else None
                    score_app = meta.get('score', 'N/A')
                    genero_app = meta.get('genero', 'Sin género')
                    installs   = meta.get('installs') or meta.get('minInstalls', 'N/A')

                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("⭐ Score", f"{score_app}" if score_app != 'N/A' else 'N/A')
                    with c2: st.metric("🏷️ Género", genero_app)
                    with c3: st.metric("📲 Instalaciones", formatear_numero(installs) if installs != 'N/A' else 'N/A')

                    st.divider()
                    col_desc, col_link = st.columns([4, 1])
                    with col_desc:
                        texto_desc = meta.get('desc_final', desc) if mostrar_desc_completa else desc
                        st.markdown(f"**Descripción:** {texto_desc}")
                    with col_link:
                        if url_app:
                            st.link_button("📲 Play Store", url_app)
                        idioma = meta.get('idioma', '') or meta.get('lang', '')
                        edad   = meta.get('edad', '') or meta.get('audience', '')
                        if idioma or edad:
                            st.caption(f"🌐 `{idioma}` · `{edad}`")

                # ── Score breakdown (ambos modos) ─────────────────────────────
                if mostrar_breakdown and breakdown:
                    st.divider()
                    st.caption("**Desglose del score:**")
                    bd_cols = st.columns(4)
                    with bd_cols[0]:
                        st.markdown(
                            f'<span class="breakdown-badge">🧠 Semántico: {breakdown.get("semantico", 0):.3f}</span>',
                            unsafe_allow_html=True
                        )
                    with bd_cols[1]:
                        st.markdown(
                            f'<span class="breakdown-badge badge-lexico">🔤 Léxico: {breakdown.get("lexico", 0):.3f}</span>',
                            unsafe_allow_html=True
                        )
                    with bd_cols[2]:
                        st.markdown(
                            f'<span class="breakdown-badge badge-metric">📊 Métricas: {breakdown.get("metricas", 0):.3f}</span>',
                            unsafe_allow_html=True
                        )
                    with bd_cols[3]:
                        st.markdown(
                            f'<span class="breakdown-badge">🎯 Raw: {breakdown.get("raw_metric", 0):.3f}</span>',
                            unsafe_allow_html=True
                        )
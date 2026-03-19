"""
app.py — Kidscorp Discovery AI
UI de testeo para búsqueda semántica de canales YouTube.
Carga bases vectoriales locales desde la carpeta 'basesTemporales/'.
"""

import streamlit as st
import os
import glob
import time
from Classes.recommendation import RecommendationEngine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kidscorp Discovery AI",
    page_icon="🚀",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tarjeta de resultado */
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
    /* Métricas más compactas */
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border-radius: 8px;
        padding: 8px 12px;
    }
    /* Score breakdown badge */
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

CARPETA_PKL = "basesTemporales"

def descubrir_segmentos() -> dict:
    """
    Escanea la carpeta basesTemporales/ y retorna un dict:
        { nombre_display: nombre_segmento }
    Solo incluye segmentos que tienen AMBOS archivos (_embeddings.pkl y _bm25.pkl).

    Esto permite que la UI solo muestre lo que realmente está disponible,
    sin hardcodear nombres de archivos.
    """
    emb_files = glob.glob(os.path.join(CARPETA_PKL, "*_embeddings.pkl"))
    segmentos = {}

    for emb_path in sorted(emb_files):
        nombre = os.path.basename(emb_path).replace("_embeddings.pkl", "")
        bm25_path = emb_path.replace("_embeddings.pkl", "_bm25.pkl")

        if os.path.exists(bm25_path):
            # Convertir nombre de archivo a display legible
            # "Youtube_English_Kids_(6-9)_Both" → "English | Kids (6-9) | Both"
            display = nombre.replace("Youtube_", "").replace("_", " ").replace("(", "(")
            segmentos[display] = nombre

    return segmentos

def formatear_numero(n) -> str:
    """Formatea números grandes con K/M para mejor legibilidad."""
    try:
        n = int(n)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)
    except:
        return "N/A"

def estado_actividad_badge(estado: str) -> str:
    """Retorna emoji + texto según el estado de actividad del canal."""
    badges = {
        "activo":       "🟢 Activo",
        "poco_activo":  "🟡 Poco activo",
        "inactivo":     "🟠 Inactivo",
        "abandonado":   "🔴 Abandonado",
        "desconocido":  "⚪ Desconocido",
    }
    return badges.get(str(estado).lower(), f"⚪ {estado}")

def tendencia_badge(tendencia: str) -> str:
    badges = {
        "creciendo":    "📈 Creciendo",
        "estable":      "➡️ Estable",
        "decayendo":    "📉 Decayendo",
        "desconocido":  "❓ Desconocido",
    }
    return badges.get(str(tendencia).lower(), f"❓ {tendencia}")

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR (cacheado — no se recrea entre reruns)
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
    st.caption("Modo: Testeo local — bases vectoriales desde `basesTemporales/`")
    st.divider()

    # ── Descubrir segmentos disponibles ──────────────────────────────────────
    segmentos_disponibles = descubrir_segmentos()

    if not segmentos_disponibles:
        st.error(f"❌ No se encontraron bases vectoriales en `{CARPETA_PKL}/`")
        st.info("Generá los PKLs con `generadorDeVectores.py` y copiálos a esa carpeta.")
        st.stop()

    st.subheader("📦 1. Seleccionar Segmento")
    st.caption(f"{len(segmentos_disponibles)} segmento(s) disponibles")

    segmento_display = st.selectbox(
        "Segmento:",
        list(segmentos_disponibles.keys())
    )
    nombre_segmento = segmentos_disponibles[segmento_display]

    if st.button("🔄 Cargar Segmento", type="primary"):
        with st.spinner(f"Cargando {segmento_display}..."):
            engine.load_from_pkl(nombre_segmento, carpeta=CARPETA_PKL)
        if engine.embeddings is not None:
            st.success(f"✅ {len(engine.df):,} canales cargados")
            if engine.metric_stats:
                st.caption(
                    f"📊 Stats: max_like_ratio={engine.metric_stats['max_like_ratio']:.2f} | "
                    f"max_videos/mes={engine.metric_stats['max_videos_por_mes']:.1f}"
                )
        else:
            st.error("❌ Error al cargar. Revisá los archivos PKL.")

    st.divider()

    # ── Pesos del score ───────────────────────────────────────────────────────
    st.subheader("⚖️ 2. Pesos del Score")
    st.caption("Deben sumar 1.0 — se normalizan automáticamente si no suman exacto.")

    w_semantic = st.slider("🧠 Semántico (E5)",  0.0, 1.0, 0.50, 0.05)
    w_lexical  = st.slider("🔤 Léxico (BM25)",   0.0, 1.0, 0.20, 0.05)
    w_metric   = st.slider("📊 Métricas",         0.0, 1.0, 0.30, 0.05)

    total_w = w_semantic + w_lexical + w_metric
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Pesos suman {total_w:.2f} — se normalizarán al buscar.")

    st.divider()

    # ── Info del segmento cargado ─────────────────────────────────────────────
    if engine.embeddings is not None and engine.source_name:
        st.subheader("ℹ️ Segmento Activo")
        st.info(
            f"**{engine.source_name}**\n\n"
            f"📺 {len(engine.df):,} canales\n\n"
            f"{'✅ Métricas disponibles' if engine.metric_stats else '⚠️ Sin métricas (PKL viejo)'}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# ÁREA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title("📺 Kidscorp Discovery AI")
st.caption("Buscador semántico de canales YouTube — modo testeo")

# ── Barra de búsqueda ─────────────────────────────────────────────────────────
col_search, col_neg = st.columns([3, 1])
with col_search:
    query = st.text_input(
        "🔎 Buscar:",
        placeholder="Ej: juegos educativos matemáticas, ciencias para niños"
    )
with col_neg:
    neg_query = st.text_input(
        "⛔ Excluir:",
        placeholder="Ej: violencia, publicidad"
    )

col_topk, col_expand = st.columns([2, 1])
with col_topk:
    top_k = st.slider("Cantidad de resultados", 1, 100, 10)
with col_expand:
    mostrar_breakdown = st.checkbox("Mostrar desglose de score", value=True)
    mostrar_desc_completa = st.checkbox("Descripción completa", value=False)

st.divider()

# ── Ejecución ─────────────────────────────────────────────────────────────────
if query:
    if engine.embeddings is None:
        st.warning("⚠️ Cargá un segmento desde el panel lateral primero.")
    else:
        start_time = time.time()

        resultados = engine.search(
            query=query,
            negative_query=neg_query if neg_query else None,
            top_k=top_k,
            semantic_weight=w_semantic,
            lexical_weight=w_lexical,
            metric_weight=w_metric,
        )

        tiempo = time.time() - start_time
        st.caption(f"⏱️ {len(resultados)} resultados en {tiempo:.3f}s — segmento: `{engine.source_name}`")

        if not resultados:
            st.info("Sin resultados. Probá con otros términos o ajustá los pesos.")

        # ── Renderizar resultados ─────────────────────────────────────────────
        for rank, item in enumerate(resultados, 1):
            score_ia   = item['score']
            meta       = item['metadata']
            titulo     = item['titulo']
            desc       = item['descripcion']
            breakdown  = item.get('score_breakdown', {})

            # Icono por score
            if score_ia > 0.65:   icono = "🔥"
            elif score_ia > 0.50: icono = "✨"
            else:                  icono = "💡"

            # URL del canal
            custom_url = meta.get('custom_url') or meta.get('channel_customurl', '')
            if custom_url:
                url_canal = custom_url if custom_url.startswith("http") else f"https://www.youtube.com/{custom_url}"
            else:
                url_canal = None

            # Métricas del canal
            subs        = formatear_numero(meta.get('subscriber_count', 0))
            views       = formatear_numero(meta.get('view_count', 0))
            avg_views   = formatear_numero(meta.get('avg_views_per_video', 0))
            videos_mes  = meta.get('videos_por_mes', 'N/A')
            like_ratio  = meta.get('avg_like_ratio', 'N/A')
            estado      = estado_actividad_badge(meta.get('estado_actividad', 'desconocido'))
            tendencia   = tendencia_badge(meta.get('tendencia', 'desconocido'))
            dias_ult    = meta.get('dias_desde_ultimo_video', 'N/A')
            pais        = meta.get('country', '')

            with st.expander(
                f"{icono} #{rank} — {titulo}  |  Score: {score_ia:.3f}",
                expanded=(rank <= 3)
            ):
                # ── Fila 1: métricas principales ─────────────────────────────
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.metric("👥 Suscriptores", subs)
                with c2: st.metric("👁️ Views totales", views)
                with c3: st.metric("📹 Avg views/video", avg_views)
                with c4: st.metric("📅 Videos/mes", f"{videos_mes}" if isinstance(videos_mes, str) else f"{videos_mes:.1f}")
                with c5: st.metric("❤️ Like ratio", f"{like_ratio}" if isinstance(like_ratio, str) else f"{like_ratio:.2f}%")

                # ── Fila 2: estado, tendencia, días ──────────────────────────
                c6, c7, c8, c9 = st.columns(4)
                with c6: st.metric("🟢 Estado", estado)
                with c7: st.metric("📈 Tendencia", tendencia)
                with c8: st.metric("📆 Días sin subir", dias_ult)
                with c9: st.metric("🌍 País", pais if pais else "N/A")

                st.divider()

                # ── Descripción + link ────────────────────────────────────────
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

                # ── Score breakdown ───────────────────────────────────────────
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
                            f'<span class="breakdown-badge">🎯 Raw métrica: {breakdown.get("raw_metric", 0):.3f}</span>',
                            unsafe_allow_html=True
                        )

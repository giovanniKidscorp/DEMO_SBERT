"""
recommendation.py
═════════════════════════════════════════════════════════════════════════════
Motor de recomendación híbrido para canales de YouTube.

ARQUITECTURA DEL SCORE FINAL
─────────────────────────────
El score final de cada canal es una combinación ponderada de tres componentes,
todos normalizados al rango [0, 1]:

    score_final = (E5 × 0.50) + (BM25 × 0.20) + (métricas × 0.30)

    • E5   (50%): Similitud semántica entre la query y el embedding del canal.
                  Captura el significado y contexto del contenido.

    • BM25 (20%): Coincidencia léxica exacta entre la query y el texto del canal.
                  Captura keywords específicas que E5 puede ignorar.

    • Métricas (30%): Score de calidad del canal basado en sus métricas numéricas.
                      Independiente de la query — mide cuán "bueno" es el canal
                      en términos de actividad, engagement, frecuencia y popularidad.

COMPOSICIÓN DEL SCORE DE MÉTRICAS
───────────────────────────────────
El score de métricas es a su vez una combinación ponderada de 4 sub-scores:

    score_metricas = (actividad × 0.35) + (engagement × 0.30)
                   + (frecuencia × 0.20) + (popularidad × 0.15)

    • Actividad   (35%): Basado en `estado_actividad` del canal.
                         activo=1.0, poco_activo=0.6, inactivo=0.2, abandonado=0.0
                         Favorece canales que publicaron recientemente.

    • Engagement  (30%): Basado en `avg_like_ratio` (likes / views × 100).
                         Normalizado con log para evitar que outliers dominen.
                         Favorece canales con buena interacción relativa al tamaño.

    • Frecuencia  (20%): Basado en `videos_por_mes`.
                         Normalizado con log, con cap en 30 videos/mes.
                         Favorece canales que publican consistentemente.

    • Popularidad (15%): Combina `subscriber_count` y `avg_views_per_video`.
                         Normalizado con log contra el dataset completo.
                         Evita que canales masivos aplasteen a medianos usando escala logarítmica.

POR QUÉ USAMOS LOG EN LAS NORMALIZACIONES
───────────────────────────────────────────
Sin log, un canal de 10M subs tiene score 1.0 y uno de 100K tiene score 0.01.
Con log, la diferencia entre 10K→100K subs es igual que entre 100K→1M.
Esto hace el ranking justo para canales de todos los tamaños.

NORMALIZACIÓN CONTRA EL DATASET
─────────────────────────────────
Los sub-scores de engagement, frecuencia y popularidad se normalizan contra
los valores máximos del dataset completo, calculados una sola vez al cargar
los datos (`_precompute_metric_stats`). Esto garantiza que el canal con el
valor más alto en cada métrica siempre tenga score 1.0.
═════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pickle
import psycopg2
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import spacy
from functools import lru_cache
from lingua import Language, LanguageDetectorBuilder
import gc

# ─────────────────────────────────────────────────────────────────────────────
# SETUP DE LENGUAJE
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGES = [Language.SPANISH, Language.ENGLISH, Language.PORTUGUESE]
LANG_DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
SUPPORTED_LANGS = {"es", "en", "pt"}

NLP_MODELS = {
    "es": spacy.load("es_core_news_sm"),
    "en": spacy.load("en_core_web_sm"),
    "pt": spacy.load("pt_core_news_sm")
}

def detect_language(text: str) -> str:
    if not text or len(text) < 3:
        return "es"
    detected = LANG_DETECTOR.detect_language_of(text)
    if detected == Language.SPANISH:    return "es"
    elif detected == Language.ENGLISH:  return "en"
    elif detected == Language.PORTUGUESE: return "pt"
    return "es"

load_dotenv()

@lru_cache(maxsize=10000)
def normalize_text(text: str):
    """Tokeniza y lematiza un texto eliminando stopwords, puntuación y números."""
    if not text:
        return []
    lang = detect_language(text)
    nlp = NLP_MODELS[lang]
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
        and len(token) > 2
    ]
    return tokens

def parsear_conceptos(query: str):
    """
    Separa la query por comas si existen, limpiando los espacios extra.
    Ej: "back to school, tennis, jockey" → ["back to school", "tennis", "jockey"]
    """
    if ',' in query:
        return [concepto.strip() for concepto in query.split(',') if concepto.strip()]
    return [query.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# HEURÍSTICA DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

# Pesos de cada sub-score dentro del score de métricas
METRIC_WEIGHTS = {
    "actividad":   0.35,
    "engagement":  0.30,
    "frecuencia":  0.20,
    "popularidad": 0.15,
}

# Mapeo de estado_actividad → score numérico
ACTIVIDAD_SCORES = {
    "activo":       1.0,
    "poco_activo":  0.6,
    "inactivo":     0.2,
    "abandonado":   0.0,
    "desconocido":  0.1,  # penalización leve por falta de datos
}

def _log_normalize(value: float, max_value: float) -> float:
    """
    Normaliza un valor usando escala logarítmica contra un máximo de referencia.

    Fórmula: log(1 + value) / log(1 + max_value)

    Por qué log: evita que valores extremos (ej. 10M subs) dominen el ranking.
    La diferencia entre 10K→100K es igual que entre 100K→1M (proporcional).

    Retorna un valor entre 0.0 y 1.0.
    """
    if max_value <= 0:
        return 0.0
    return np.log1p(value) / np.log1p(max_value)


def calcular_score_metricas(row: dict, stats: dict) -> float:
    """
    Calcula el score de métricas de un canal, normalizado entre 0 y 1.

    Parámetros
    ──────────
    row   : dict con los campos de metadata del canal (viene del DataFrame)
    stats : dict con los valores máximos del dataset, pre-computados una vez
            al cargar los datos. Contiene:
              - max_like_ratio     : máximo avg_like_ratio del dataset
              - max_videos_por_mes : máximo videos_por_mes del dataset
              - max_popularidad    : máximo del score combinado sub×vistas del dataset

    Retorna
    ───────
    float entre 0.0 y 1.0
    """

    # ── SUB-SCORE 1: ACTIVIDAD (35%) ────────────────────────────────────────
    # Usa el campo `estado_actividad` que viene del scraper.
    # Valores posibles: activo, poco_activo, inactivo, abandonado, desconocido
    estado = str(row.get("estado_actividad", "desconocido")).lower()
    s_actividad = ACTIVIDAD_SCORES.get(estado, 0.1)

    # ── SUB-SCORE 2: ENGAGEMENT (30%) ───────────────────────────────────────
    # avg_like_ratio = (total_likes / total_views) × 100
    # Mide interacción relativa — un canal pequeño con muchos likes por view
    # puede superar a uno grande con pocos.
    like_ratio = float(row.get("avg_like_ratio") or 0)
    s_engagement = _log_normalize(like_ratio, stats["max_like_ratio"])

    # ── SUB-SCORE 3: FRECUENCIA (20%) ────────────────────────────────────────
    # videos_por_mes = promedio de videos publicados en los últimos 3 meses / 3
    # Cap en 30 videos/mes para no penalizar canales normales vs. factories de contenido.
    videos_mes = float(row.get("videos_por_mes") or 0)
    videos_mes_capped = min(videos_mes, 30.0)
    max_frecuencia_capped = min(stats["max_videos_por_mes"], 30.0)
    s_frecuencia = _log_normalize(videos_mes_capped, max_frecuencia_capped)

    # ── SUB-SCORE 4: POPULARIDAD (15%) ───────────────────────────────────────
    # Combina subscriber_count y avg_views_per_video en un único valor.
    # Fórmula: sqrt(subs × avg_views) — la raíz cuadrada balancea ambos factores.
    # Luego se normaliza con log contra el máximo del dataset.
    subs      = float(row.get("subscriber_count") or 0)
    avg_views = float(row.get("avg_views_per_video") or 0)
    popularidad_raw = np.sqrt(subs * avg_views) if (subs > 0 and avg_views > 0) else 0.0
    s_popularidad = _log_normalize(popularidad_raw, stats["max_popularidad"])

    # ── SCORE FINAL DE MÉTRICAS ───────────────────────────────────────────────
    score = (
        s_actividad   * METRIC_WEIGHTS["actividad"]   +
        s_engagement  * METRIC_WEIGHTS["engagement"]  +
        s_frecuencia  * METRIC_WEIGHTS["frecuencia"]  +
        s_popularidad * METRIC_WEIGHTS["popularidad"]
    )

    return float(np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0))


def precompute_metric_stats(df: pd.DataFrame) -> dict:
    """
    Pre-computa los valores máximos del dataset necesarios para normalizar
    los sub-scores de métricas. Se llama una sola vez al cargar los datos.

    Por qué pre-computar: evita recalcular max() en cada búsqueda.
    Con 40K canales, recalcular en cada query agregaría ~50ms innecesarios.

    Retorna un dict con:
        max_like_ratio     : máximo avg_like_ratio del dataset
        max_videos_por_mes : máximo videos_por_mes del dataset
        max_popularidad    : máximo del score combinado sqrt(subs × avg_views)
    """
    def safe_max(series):
        """Retorna el máximo ignorando NaN y valores negativos."""
        clean = pd.to_numeric(series, errors='coerce').dropna()
        clean = clean[clean >= 0]
        return float(clean.max()) if len(clean) > 0 else 1.0

    max_like_ratio     = safe_max(df.get("avg_like_ratio",     pd.Series(dtype=float)))
    max_videos_por_mes = safe_max(df.get("videos_por_mes",     pd.Series(dtype=float)))

    # Popularidad combinada: sqrt(subs × avg_views)
    subs      = pd.to_numeric(df.get("subscriber_count",    pd.Series(dtype=float)), errors='coerce').fillna(0)
    avg_views = pd.to_numeric(df.get("avg_views_per_video", pd.Series(dtype=float)), errors='coerce').fillna(0)
    popularidad_series = np.sqrt(subs * avg_views)
    max_popularidad = float(popularidad_series.max()) if len(popularidad_series) > 0 else 1.0

    stats = {
        "max_like_ratio":     max(max_like_ratio,     1.0),
        "max_videos_por_mes": max(max_videos_por_mes, 1.0),
        "max_popularidad":    max(max_popularidad,    1.0),
    }

    print(f"   📊 Stats de métricas pre-computadas:")
    print(f"      max_like_ratio:     {stats['max_like_ratio']:.4f}")
    print(f"      max_videos_por_mes: {stats['max_videos_por_mes']:.2f}")
    print(f"      max_popularidad:    {stats['max_popularidad']:.0f}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Motor de recomendación híbrido: E5 (semántico) + BM25 (léxico) + Métricas.

    Pesos por defecto:
        semantic_weight = 0.50
        lexical_weight  = 0.20
        metric_weight   = 0.30
    """

    def __init__(self, model_name='intfloat/multilingual-e5-base'):
        print(f"🧠 Cargando Modelo E5 (Solo para queries): {model_name}...")
        self.model = SentenceTransformer(model_name)

        self.df           = pd.DataFrame()
        self.embeddings   = None
        self.bm25         = None
        self.source_name  = ""
        self.metric_stats = {}  # se llena al cargar datos

    # ─────────────────────────────────────────────────────────────────────────
    # CARGA DE DATOS
    # ─────────────────────────────────────────────────────────────────────────

    def _load_from_db(self, fuente):
        """
        Carga datos desde PostgreSQL con optimización para datasets grandes (>10K).
        Usa procesamiento por lotes para evitar saturar la RAM.
        Al finalizar, pre-computa las stats de métricas.
        """
        print(f"\n☁️ Conectando a Postgres para cargar: '{fuente}'")
        start = time.time()

        conn = None
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                port="5432",
                sslmode="require"
            )
            cursor = conn.cursor()

            # ── 1. CARGAR BM25 ────────────────────────────────────────────
            print("   📥 Descargando índice léxico (BM25)...")
            cursor.execute(
                "SELECT archivo_pickle FROM keywordsearch.archivos_bm25 WHERE fuente = %s",
                (fuente,)
            )
            row = cursor.fetchone()
            if row:
                self.bm25 = pickle.loads(row[0])
                print("   ✅ BM25 cargado y listo en memoria.")
            else:
                print(f"   ❌ ERROR: No se encontró BM25 para '{fuente}'.")

            # ── 2. CONTAR REGISTROS ───────────────────────────────────────
            print("   📊 Contando registros...")
            cursor.execute(
                "SELECT COUNT(*) FROM keywordsearch.vectores_e5 WHERE fuente = %s",
                (fuente,)
            )
            total_rows = cursor.fetchone()[0]
            print(f"   📊 Total de registros: {total_rows:,}")

            estimated_mb = (total_rows * 3072) / (1024 * 1024)
            print(f"   💾 Memoria estimada para embeddings: {estimated_mb:.1f} MB")

            # ── 3. ESTRATEGIA DE CARGA ────────────────────────────────────
            if total_rows <= 10000:
                print("   ⚡ Dataset pequeño: Carga directa")
                self._load_direct(cursor, fuente, total_rows)
            else:
                print("   🔄 Dataset grande: Carga por lotes")
                self._load_batched(cursor, fuente, total_rows)

            cursor.close()
            gc.collect()

            # ── 4. PRE-COMPUTAR STATS DE MÉTRICAS ─────────────────────────
            # Se hace aquí, una sola vez, para no recalcular en cada búsqueda.
            if not self.df.empty:
                print("   ⚙️ Pre-computando stats de métricas...")
                self.metric_stats = precompute_metric_stats(self.df)

        except Exception as e:
            print(f"❌ Error de conexión a DB: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

        print(f"⏱️ Tiempo total de descarga: {time.time() - start:.2f}s\n")

    def _load_direct(self, cursor, fuente, total_rows):
        """Carga directa para datasets pequeños (<10K registros)."""
        cursor.execute("""
            SELECT metadata, embedding::text
            FROM keywordsearch.vectores_e5
            WHERE fuente = %s
            ORDER BY id ASC
        """, (fuente,))
        rows = cursor.fetchall()

        if rows:
            df_records = []
            tensor_list = []
            for meta, emb_str in rows:
                df_records.append(meta)
                arr = np.fromstring(emb_str[1:-1], sep=',')
                tensor_list.append(arr)
            self.df = pd.DataFrame(df_records)
            self.embeddings = torch.tensor(np.array(tensor_list), dtype=torch.float32)
            print(f"   ✅ {len(self.df)} registros cargados")
        else:
            print(f"   ❌ ERROR: No se encontraron vectores para '{fuente}'.")

    def _load_batched(self, cursor, fuente, total_rows):
        """
        Carga por lotes para datasets grandes (>10K registros).
        Procesa en chunks de 5,000 registros para no saturar memoria.
        """
        BATCH_SIZE = 5000
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   📦 Procesando en {num_batches} lotes de {BATCH_SIZE:,} registros...")

        df_records = []
        embeddings_arrays = []

        for batch_num in range(num_batches):
            offset = batch_num * BATCH_SIZE
            print(
                f"   ⏳ Lote {batch_num + 1}/{num_batches} "
                f"(registros {offset:,} - {min(offset + BATCH_SIZE, total_rows):,})...",
                end='', flush=True
            )
            cursor.execute("""
                SELECT metadata, embedding::text
                FROM keywordsearch.vectores_e5
                WHERE fuente = %s
                ORDER BY id ASC
                LIMIT %s OFFSET %s
            """, (fuente, BATCH_SIZE, offset))

            batch_rows = cursor.fetchall()
            if not batch_rows:
                print(" ⚠️ Vacío, saliendo.")
                break

            batch_start = time.time()
            for meta, emb_str in batch_rows:
                df_records.append(meta)
                arr = np.fromstring(emb_str[1:-1], sep=',', dtype=np.float32)
                embeddings_arrays.append(arr)

            print(f" ✓ ({time.time() - batch_start:.1f}s)")
            del batch_rows
            gc.collect()

        print("   🔧 Consolidando datos...")
        self.df = pd.DataFrame(df_records)
        print("   🔧 Construyendo tensor de embeddings...")
        embeddings_np = np.vstack(embeddings_arrays)
        self.embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
        print(f"   ✅ {len(self.df):,} registros cargados y sincronizados.")

        del df_records, embeddings_arrays, embeddings_np
        gc.collect()

    def load_from_postgres(self, db_config=None, limit=None, force_refresh=False):
        self.source_name = "youtube_channels_db"
        self._load_from_db(self.source_name)

    def load_from_json(self, json_path, force_refresh=False):
        """Carga desde Postgres usando el nombre del JSON como identificador de fuente."""
        fuente = os.path.basename(json_path).replace('.json', '')
        self.source_name = fuente
        self._load_from_db(fuente)

    def load_from_pkl(self, nombre_segmento: str, carpeta: str = "basesTemporales"):
        """
        Carga embeddings + metadata + BM25 desde archivos PKL locales.

        Espera encontrar en `carpeta` dos archivos con el formato:
            {nombre_segmento}_embeddings.pkl  → dict con 'embeddings' y 'metadata'
            {nombre_segmento}_bm25.pkl        → índice BM25Okapi

        Parámetros
        ──────────
        nombre_segmento : nombre del segmento sin extensión.
                          Ej: "Youtube_English_Kids_(6-9)_Both"
        carpeta         : carpeta donde están los PKLs (default: "basesTemporales")

        Por qué dos archivos separados:
            Los embeddings y la metadata van juntos porque deben estar sincronizados
            por índice — el canal en la posición 0 del tensor es el mismo que el
            registro 0 de metadata. El BM25 se guarda aparte porque es un objeto
            independiente que no necesita estar alineado por índice.
        """
        ruta_emb  = os.path.join(carpeta, f"{nombre_segmento}_embeddings.pkl")
        ruta_bm25 = os.path.join(carpeta, f"{nombre_segmento}_bm25.pkl")

        print(f"\n📂 Cargando segmento desde PKL local: '{nombre_segmento}'")
        start = time.time()

        # ── Cargar embeddings + metadata ──────────────────────────────────────
        if not os.path.exists(ruta_emb):
            print(f"   ❌ No se encontró: {ruta_emb}")
            return
        with open(ruta_emb, 'rb') as f:
            data = pickle.load(f)

        # Soporte para PKL viejo (solo tensor) y nuevo (dict con metadata)
        if isinstance(data, dict):
            self.embeddings = data["embeddings"]
            self.df = pd.DataFrame(data["metadata"])
        else:
            # PKL generado antes del cambio de formato — solo tiene el tensor
            print("   ⚠️ PKL sin metadata (formato viejo). Las métricas no estarán disponibles.")
            self.embeddings = data
            self.df = pd.DataFrame()

        print(f"   ✅ Embeddings cargados: {self.embeddings.shape[0]:,} canales")

        # ── Cargar BM25 ───────────────────────────────────────────────────────
        if not os.path.exists(ruta_bm25):
            print(f"   ❌ No se encontró: {ruta_bm25}")
            return
        with open(ruta_bm25, 'rb') as f:
            self.bm25 = pickle.load(f)
        print(f"   ✅ BM25 cargado.")

        self.source_name = nombre_segmento

        # ── Pre-computar stats de métricas ────────────────────────────────────
        if not self.df.empty:
            print("   ⚙️ Pre-computando stats de métricas...")
            self.metric_stats = precompute_metric_stats(self.df)
        else:
            self.metric_stats = {}
            print("   ⚠️ Sin metadata — score de métricas desactivado.")

        print(f"⏱️ Tiempo de carga: {time.time() - start:.2f}s\n")

    # ─────────────────────────────────────────────────────────────────────────
    # BÚSQUEDA
    # ─────────────────────────────────────────────────────────────────────────

    def search(self,
               query,
               negative_query=None,
               top_k=5,
               filters=None,
               semantic_weight=0.50,
               lexical_weight=0.20,
               metric_weight=0.30,
               hard_negative_filter=True,
               bm25_negative_penalty=0.7):
        """
        Busca canales combinando tres señales: E5, BM25 y Métricas.

        Parámetros
        ──────────
        query               : texto de búsqueda (puede tener comas para múltiples conceptos)
        negative_query      : concepto a penalizar/excluir
        top_k               : cantidad de resultados a retornar
        filters             : dict con filtros duros (ej. {'score_min': 4.0})
        semantic_weight     : peso del score E5 (default 0.50)
        lexical_weight      : peso del score BM25 (default 0.20)
        metric_weight       : peso del score de métricas (default 0.30)
        hard_negative_filter: si True, excluye canales con keywords negativas en su texto
        bm25_negative_penalty: penalización BM25 para canales con contenido negativo (0-1)

        Retorna
        ───────
        Lista de dicts con: score, score_breakdown, titulo, descripcion, metadata
        score_breakdown desglosa la contribución de cada componente para debugging.
        """

        if self.embeddings is None or self.bm25 is None:
            return []

        # Normalizar pesos para que sumen 1.0
        total_weight = semantic_weight + lexical_weight + metric_weight
        if abs(total_weight - 1.0) > 0.01:
            semantic_weight = semantic_weight / total_weight
            lexical_weight  = lexical_weight  / total_weight
            metric_weight   = metric_weight   / total_weight

        # ── 1. SCORE SEMÁNTICO (E5) ───────────────────────────────────────────
        # E5 espera el prefijo "query: " para consultas y "passage: " para documentos.
        conceptos = parsear_conceptos(query)
        query_text = "query: " + ", ".join(conceptos)
        query_vec = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)

        # Negative query: se resta del vector de búsqueda para alejar resultados no deseados.
        # El factor 0.8 evita que el negativo domine completamente la dirección del vector.
        if negative_query:
            neg_conceptos = parsear_conceptos(negative_query)
            neg_text = "query: " + ", ".join(neg_conceptos)
            neg_vec = self.model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
            query_vec = query_vec - (neg_vec * 0.8)

        # candidate_size: buscamos más candidatos que top_k para que el re-ranking
        # por BM25 y métricas tenga margen de maniobra.
        candidate_size = max(1000, top_k * 10, int(len(self.df) * 0.1))
        candidate_size = min(candidate_size, len(self.df))

        semantic_hits = util.semantic_search(query_vec, self.embeddings, top_k=candidate_size)

        # ── 2. SCORE LÉXICO (BM25) ────────────────────────────────────────────
        bm25_tokens = []
        for c in conceptos:
            bm25_tokens.extend(normalize_text(c))
            if ' ' in c:
                # Versión fusionada: "back to school" → "backtoschool"
                bm25_tokens.append(c.replace(' ', '').lower())

        tokenized_query = list(dict.fromkeys(bm25_tokens))  # deduplicar
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Penalización BM25 para contenido negativo
        negative_keywords = set()
        if negative_query:
            neg_conceptos = parsear_conceptos(negative_query)
            neg_tokens = []
            for c in neg_conceptos:
                neg_tokens.extend(normalize_text(c))
                if ' ' in c:
                    neg_tokens.append(c.replace(' ', '').lower())
            negative_keywords = set(neg_tokens)

            if bm25_negative_penalty > 0:
                for idx in range(len(bm25_scores)):
                    if idx >= len(self.df):
                        continue
                    row = self.df.iloc[idx]
                    texto = str(row.get('channel_title', '')) + " " + str(row.get('channel_description', ''))
                    if any(keyword in texto.lower() for keyword in negative_keywords):
                        bm25_scores[idx] *= (1 - bm25_negative_penalty)

        # Normalizar BM25 al rango [0, 1] dividiendo por el máximo del dataset
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        # ── 3. FUSIÓN DE SCORES ───────────────────────────────────────────────
        # Estructura: {idx: {"semantic": float, "lexical": float, "metric": float}}
        score_components = {}

        # Agregar scores semánticos
        for hit in semantic_hits[0]:
            idx = hit['corpus_id']
            if idx < len(self.df):
                score_components[idx] = {
                    "semantic": float(hit['score']),
                    "lexical":  0.0,
                    "metric":   0.0
                }

        # Agregar scores BM25
        for idx, bm25_score in enumerate(bm25_scores):
            if idx >= len(self.df):
                continue
            normalized_bm25 = bm25_score / max_bm25
            if idx in score_components:
                score_components[idx]["lexical"] = normalized_bm25
            else:
                score_components[idx] = {"semantic": 0.0, "lexical": normalized_bm25, "metric": 0.0}

        # Agregar scores de métricas
        # Solo se calculan para candidatos que ya pasaron E5 o BM25, no para todo el dataset.
        # Esto es eficiente: con 40K canales calcular métricas para todos sería innecesario.
        if self.metric_stats:
            for idx in score_components:
                row = self.df.iloc[idx]
                score_components[idx]["metric"] = calcular_score_metricas(
                    row.to_dict(), self.metric_stats
                )

        # Score final ponderado
        combined_scores = {
            idx: (
                components["semantic"] * semantic_weight +
                components["lexical"]  * lexical_weight  +
                components["metric"]   * metric_weight
            )
            for idx, components in score_components.items()
        }

        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # ── 4. CONSTRUIR RESULTADOS ───────────────────────────────────────────
        results = []

        for idx, combined_score in sorted_indices:
            if len(results) >= top_k:
                break

            row = self.df.iloc[idx]

            # Filtro duro de negativos: excluye canales que mencionan el concepto negativo
            if hard_negative_filter and negative_keywords:
                texto_para_filtrar = " ".join([
                    str(row.get('channel_title', '')),
                    str(row.get('channel_description', '')),
                    str(row.get('channel_bs_ch_keywords', ''))
                ])
                if any(keyword in normalize_text(texto_para_filtrar) for keyword in negative_keywords):
                    continue

            # Filtros numéricos opcionales (ej. score_min para apps)
            if filters:
                if 'score_min' in filters and float(row.get('score', 0)) < filters['score_min']:
                    continue
                if 'genero' in filters and row.get('genero') != filters['genero']:
                    continue

            title = row.get('channel_title') or row.get('common_title') or "Sin Título"
            desc  = row.get('channel_description') or row.get('desc_final') or ""

            components = score_components.get(idx, {})

            results.append({
                # Score final combinado
                "score": float(combined_score),

                # Desglose por componente — útil para debugging y transparencia
                # Cada valor muestra la contribución PONDERADA al score final
                "score_breakdown": {
                    "semantico":  round(components.get("semantic", 0) * semantic_weight, 4),
                    "lexico":     round(components.get("lexical",  0) * lexical_weight,  4),
                    "metricas":   round(components.get("metric",   0) * metric_weight,   4),
                    # Sub-scores de métricas (sin ponderar, para referencia)
                    "raw_metric": round(components.get("metric", 0), 4),
                },

                "titulo":      title,
                "descripcion": desc[:200] + ("..." if len(desc) > 200 else ""),
                "metadata":    row.to_dict()
            })

        return results
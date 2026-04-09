"""
recommendation.py
═════════════════════════════════════════════════════════════════════════════
Motor de recomendación híbrido: E5 (semántico) + BM25 (léxico) + Métricas.

ARQUITECTURA DEL SCORE FINAL
─────────────────────────────
    score_final = (E5 × 0.50) + (BM25 × 0.20) + (métricas × 0.30)

    • E5   (50%): Similitud semántica entre la query y el embedding.
    • BM25 (20%): Coincidencia léxica exacta.
    • Métricas (30%): Score de calidad basado en métricas numéricas.
                      Solo aplica a YouTube — para apps se ignora (peso=0).

COMPOSICIÓN DEL SCORE DE MÉTRICAS (YouTube)
─────────────────────────────────────────────
    score_metricas = (actividad × 0.35) + (engagement × 0.30)
                   + (frecuencia × 0.20) + (popularidad × 0.15)

MODOS DE CARGA
───────────────
    • load_from_pkl()    : carga desde archivos PKL locales (YouTube y Apps)
    • load_from_postgres(): carga desde PostgreSQL
    • _load_from_db()    : interno, usado por los dos anteriores para Postgres

DETECCIÓN DE TIPO DE SEGMENTO
───────────────────────────────
    El engine detecta automáticamente si el segmento es de YouTube o Apps
    basándose en el nombre del segmento (source_name). Esto afecta:
    - Qué campo usar como título (channel_title vs common_title)
    - Si aplicar o no el score de métricas
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
from deep_translator import GoogleTranslator
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
    if detected == Language.SPANISH:      return "es"
    elif detected == Language.ENGLISH:    return "en"
    elif detected == Language.PORTUGUESE: return "pt"
    return "es"

load_dotenv()


def expand_token_variants(token: str) -> list:
    """
    Genera variantes ortográficas de un token.
    'make-up' → ['makeup', 'make-up', 'make up']
    'makeup'  → ['makeup', 'make-up']  (si detecta camelCase o compound)
    """
    variants = {token}  # siempre incluir el original

    # Si tiene guión: agregar versión junta y separada
    if '-' in token:
        variants.add(token.replace('-', ''))    # make-up → makeup
        variants.add(token.replace('-', ' '))    # make-up → make up

    # Si NO tiene guión ni espacio, pero es una palabra compuesta conocida,
    # podrías agregar variante con guión (opcional, más agresivo)
    # Ej: "makeup" → también buscar "make-up"
    # Esto se puede hacer con un diccionario o heurística simple:
    if '-' not in token and ' ' not in token and len(token) > 5:
        # Heurística: intentar partir en subpalabras comunes
        # (esto es opcional, el caso más importante es el de arriba)
        pass

    return list(variants)


@lru_cache(maxsize=10000)
def normalize_text(text: str):
    """Tokeniza y lematiza un texto eliminando stopwords, puntuación y números."""
    if not text:
        return []
    lang = detect_language(text)
    nlp = NLP_MODELS[lang]
    doc = nlp(text.lower())

    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or len(token) < 3:
            continue

        lemma = token.lemma_
        # Expandir variantes para cada lemma
        for variant in expand_token_variants(lemma):
            if len(variant) > 2:
                tokens.append(variant)

    # También procesar el texto raw para capturar tokens que spaCy no maneje bien
    raw_tokens = text.lower().split()
    for rt in raw_tokens:
        if '-' in rt:
            for variant in expand_token_variants(rt):
                if len(variant) > 2 and variant not in tokens:
                    tokens.append(variant)

    return tokens

def parsear_conceptos(query: str):
    """
    Separa la query por comas.
    Ej: "back to school, tennis" → ["back to school", "tennis"]
    """
    if ',' in query:
        return [c.strip() for c in query.split(',') if c.strip()]
    return [query.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# HEURÍSTICA DE MÉTRICAS (solo YouTube)
# ─────────────────────────────────────────────────────────────────────────────

METRIC_WEIGHTS = {
    "actividad":   0.35,
    "engagement":  0.30,
    "frecuencia":  0.20,
    "popularidad": 0.15,
}

ACTIVIDAD_SCORES = {
    "activo":      1.0,
    "poco_activo": 0.6,
    "inactivo":    0.2,
    "abandonado":  0.0,
    "desconocido": 0.1,
}

def _log_normalize(value: float, max_value: float) -> float:
    """
    Normaliza con escala logarítmica: log(1+v) / log(1+max).
    Evita que valores extremos dominen el ranking.
    """
    if max_value <= 0:
        return 0.0
    return np.log1p(value) / np.log1p(max_value)


def calcular_score_metricas(row: dict, stats: dict) -> float:
    """
    Calcula el score de métricas de un canal YouTube, normalizado [0, 1].
    No se llama para apps — se devuelve 0.0 directamente en ese caso.
    """
    estado      = str(row.get("estado_actividad", "desconocido")).lower()
    s_actividad = ACTIVIDAD_SCORES.get(estado, 0.1)

    like_ratio   = float(row.get("avg_like_ratio") or 0)
    s_engagement = _log_normalize(like_ratio, stats["max_like_ratio"])

    videos_mes            = float(row.get("videos_por_mes") or 0)
    videos_mes_capped     = min(videos_mes, 30.0)
    max_frecuencia_capped = min(stats["max_videos_por_mes"], 30.0)
    s_frecuencia          = _log_normalize(videos_mes_capped, max_frecuencia_capped)

    subs            = float(row.get("subscriber_count") or 0)
    avg_views       = float(row.get("avg_views_per_video") or 0)
    popularidad_raw = np.sqrt(subs * avg_views) if (subs > 0 and avg_views > 0) else 0.0
    s_popularidad   = _log_normalize(popularidad_raw, stats["max_popularidad"])

    score = (
        s_actividad   * METRIC_WEIGHTS["actividad"]   +
        s_engagement  * METRIC_WEIGHTS["engagement"]  +
        s_frecuencia  * METRIC_WEIGHTS["frecuencia"]  +
        s_popularidad * METRIC_WEIGHTS["popularidad"]
    )
    return float(np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0))


def precompute_metric_stats(df: pd.DataFrame) -> dict:
    """
    Pre-computa los máximos del dataset para normalizar métricas.
    Se llama una sola vez al cargar datos de YouTube.
    """
    def safe_max(series):
        clean = pd.to_numeric(series, errors='coerce').dropna()
        clean = clean[clean >= 0]
        return float(clean.max()) if len(clean) > 0 else 1.0

    max_like_ratio     = safe_max(df.get("avg_like_ratio",     pd.Series(dtype=float)))
    max_videos_por_mes = safe_max(df.get("videos_por_mes",     pd.Series(dtype=float)))

    subs      = pd.to_numeric(df.get("subscriber_count",    pd.Series(dtype=float)), errors='coerce').fillna(0)
    avg_views = pd.to_numeric(df.get("avg_views_per_video", pd.Series(dtype=float)), errors='coerce').fillna(0)
    max_popularidad = float(np.sqrt(subs * avg_views).max()) if len(subs) > 0 else 1.0

    stats = {
        "max_like_ratio":     max(max_like_ratio,     1.0),
        "max_videos_por_mes": max(max_videos_por_mes, 1.0),
        "max_popularidad":    max(max_popularidad,    1.0),
    }

    print(f"   📊 Stats métricas: like_ratio={stats['max_like_ratio']:.2f} | "
          f"videos/mes={stats['max_videos_por_mes']:.1f} | "
          f"popularidad={stats['max_popularidad']:.0f}")

    return stats


def es_segmento_youtube(source_name: str) -> bool:
    """
    Detecta si el segmento es de YouTube o de Apps basándose en el nombre.
    Los segmentos de YouTube empiezan con 'Youtube_'.
    Los de apps empiezan con 'mp.' o 'Apps_' o similares.
    """
    name_lower = source_name.lower()
    return name_lower.startswith("youtube") or "youtube" in name_lower


def get_titulo(row: pd.Series, es_youtube: bool) -> str:
    """
    Obtiene el título correcto según el tipo de segmento.
    - YouTube: channel_title
    - Apps: common_title, app_name, title (en ese orden de preferencia)
    """
    if es_youtube:
        return row.get('channel_title') or row.get('common_title') or "Sin Título"
    else:
        return (row.get('common_title') or row.get('titulo_store') or
                row.get('title') or row.get('channel_title') or "Sin Título")


def get_descripcion(row: pd.Series, es_youtube: bool) -> str:
    """Obtiene la descripción correcta según el tipo de segmento."""
    if es_youtube:
        return row.get('channel_description') or row.get('desc_final') or ""
    else:
        return row.get('desc_corta') or row.get('description') or ""


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Motor de recomendación híbrido: E5 + BM25 + Métricas.
    Soporta segmentos de YouTube (PKL local) y Apps (Postgres o PKL).
    """

    def __init__(self, model_name='intfloat/multilingual-e5-base'):
        print(f"🧠 Cargando Modelo E5: {model_name}...")
        self.model = SentenceTransformer(model_name)

        self.df           = pd.DataFrame()
        self.embeddings   = None
        self.bm25         = None
        self.source_name  = ""
        self.metric_stats = {}
        self.es_youtube   = False  # se actualiza al cargar cada segmento

    # ─────────────────────────────────────────────────────────────────────────
    # CARGA DESDE PKL LOCAL
    # ─────────────────────────────────────────────────────────────────────────

    def load_from_pkl(self, nombre_segmento: str, carpeta: str = "basesTemporales"):
        """
        Carga embeddings + metadata + BM25 desde archivos PKL locales.

        Espera:
            {carpeta}/{nombre_segmento}_embeddings.pkl  → dict con 'embeddings' y 'metadata'
            {carpeta}/{nombre_segmento}_bm25.pkl        → índice BM25Okapi

        Compatible con PKLs viejos (solo tensor, sin metadata).
        Detecta automáticamente si es YouTube o Apps por el nombre.
        """
        ruta_emb  = os.path.join(carpeta, f"{nombre_segmento}_embeddings.pkl")
        ruta_bm25 = os.path.join(carpeta, f"{nombre_segmento}_bm25.pkl")

        print(f"\n📂 Cargando: '{nombre_segmento}'")
        start = time.time()

        if not os.path.exists(ruta_emb):
            print(f"   ❌ No se encontró: {ruta_emb}")
            return

        with open(ruta_emb, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.embeddings = data["embeddings"]
            self.df = pd.DataFrame(data["metadata"])
        else:
            print("   ⚠️ PKL sin metadata (formato viejo). Métricas desactivadas.")
            self.embeddings = data
            self.df = pd.DataFrame()

        print(f"   ✅ {self.embeddings.shape[0]:,} registros cargados")

        if not os.path.exists(ruta_bm25):
            print(f"   ❌ No se encontró: {ruta_bm25}")
            return

        with open(ruta_bm25, 'rb') as f:
            self.bm25 = pickle.load(f)
        print(f"   ✅ BM25 cargado")

        self.source_name = nombre_segmento
        self.es_youtube  = es_segmento_youtube(nombre_segmento)

        print(f"   🎯 Tipo: {'📺 YouTube' if self.es_youtube else '📱 Apps'}")

        # Métricas solo para YouTube
        if self.es_youtube and not self.df.empty:
            print("   ⚙️ Pre-computando stats de métricas...")
            self.metric_stats = precompute_metric_stats(self.df)
        else:
            self.metric_stats = {}
            if not self.es_youtube:
                print("   ℹ️ Apps: score de métricas desactivado")

        print(f"⏱️ Carga: {time.time() - start:.2f}s\n")

    # ─────────────────────────────────────────────────────────────────────────
    # CARGA DESDE POSTGRES
    # ─────────────────────────────────────────────────────────────────────────

    def _load_from_db(self, fuente):
        """
        Carga datos desde PostgreSQL.
        Detecta automáticamente si es YouTube o Apps por el nombre de fuente.
        """
        print(f"\n☁️ Conectando a Postgres: '{fuente}'")
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

            print("   📥 Descargando BM25...")
            cursor.execute(
                "SELECT archivo_pickle FROM keywordsearch.archivos_bm25 WHERE fuente = %s",
                (fuente,)
            )
            row = cursor.fetchone()
            if row:
                self.bm25 = pickle.loads(row[0])
                print("   ✅ BM25 listo.")
            else:
                print(f"   ❌ No se encontró BM25 para '{fuente}'.")

            cursor.execute(
                "SELECT COUNT(*) FROM keywordsearch.vectores_e5 WHERE fuente = %s",
                (fuente,)
            )
            total_rows = cursor.fetchone()[0]
            print(f"   📊 {total_rows:,} registros")

            if total_rows <= 10000:
                self._load_direct(cursor, fuente, total_rows)
            else:
                self._load_batched(cursor, fuente, total_rows)

            cursor.close()
            gc.collect()

            self.source_name = fuente
            self.es_youtube  = es_segmento_youtube(fuente)

            if self.es_youtube and not self.df.empty:
                self.metric_stats = precompute_metric_stats(self.df)
            else:
                self.metric_stats = {}

        except Exception as e:
            print(f"❌ Error DB: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

        print(f"⏱️ Descarga: {time.time() - start:.2f}s\n")

    def _load_direct(self, cursor, fuente, total_rows):
        cursor.execute("""
            SELECT metadata, embedding::text
            FROM keywordsearch.vectores_e5
            WHERE fuente = %s ORDER BY id ASC
        """, (fuente,))
        rows = cursor.fetchall()
        if rows:
            df_records, tensor_list = [], []
            for meta, emb_str in rows:
                df_records.append(meta)
                tensor_list.append(np.fromstring(emb_str[1:-1], sep=','))
            self.df = pd.DataFrame(df_records)
            self.embeddings = torch.tensor(np.array(tensor_list), dtype=torch.float32)
            print(f"   ✅ {len(self.df)} registros cargados")

    def _load_batched(self, cursor, fuente, total_rows):
        BATCH_SIZE = 5000
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   📦 {num_batches} lotes de {BATCH_SIZE:,}...")

        df_records, embeddings_arrays = [], []
        for batch_num in range(num_batches):
            offset = batch_num * BATCH_SIZE
            print(f"   ⏳ Lote {batch_num+1}/{num_batches}...", end='', flush=True)
            cursor.execute("""
                SELECT metadata, embedding::text
                FROM keywordsearch.vectores_e5
                WHERE fuente = %s ORDER BY id ASC LIMIT %s OFFSET %s
            """, (fuente, BATCH_SIZE, offset))
            batch_rows = cursor.fetchall()
            if not batch_rows:
                print(" ⚠️ Vacío.")
                break
            t = time.time()
            for meta, emb_str in batch_rows:
                df_records.append(meta)
                embeddings_arrays.append(np.fromstring(emb_str[1:-1], sep=',', dtype=np.float32))
            print(f" ✓ ({time.time()-t:.1f}s)")
            del batch_rows
            gc.collect()

        self.df = pd.DataFrame(df_records)
        self.embeddings = torch.tensor(np.vstack(embeddings_arrays), dtype=torch.float32)
        print(f"   ✅ {len(self.df):,} registros listos.")
        del df_records, embeddings_arrays
        gc.collect()

    def load_from_postgres(self, db_config=None, limit=None, force_refresh=False):
        self.source_name = "youtube_channels_db"
        self._load_from_db(self.source_name)

    def load_from_json(self, json_path, force_refresh=False):
        fuente = os.path.basename(json_path).replace('.json', '')
        self.source_name = fuente
        self._load_from_db(fuente)

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
        Busca combinando E5 + BM25 + Métricas.

        Para Apps: metric_weight se ignora automáticamente (se redistribuye
        entre semántico y léxico para mantener los pesos relativos).

        Retorna lista de dicts con: score, score_breakdown, titulo, descripcion, metadata.
        """
        if self.embeddings is None or self.bm25 is None:
            return []

        # Para apps, desactivar métricas redistribuyendo su peso
        if not self.es_youtube:
            total = semantic_weight + lexical_weight
            if total > 0:
                semantic_weight = semantic_weight / total
                lexical_weight  = lexical_weight  / total
            metric_weight = 0.0
        else:
            total_weight = semantic_weight + lexical_weight + metric_weight
            if abs(total_weight - 1.0) > 0.01:
                semantic_weight /= total_weight
                lexical_weight  /= total_weight
                metric_weight   /= total_weight

        # ── 1. SCORE SEMÁNTICO (E5) ───────────────────────────────────────────
        conceptos  = parsear_conceptos(query)
        query_text = "query: " + ", ".join(conceptos)
        query_vec  = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)

        if negative_query:
            neg_text = "query: " + ", ".join(parsear_conceptos(negative_query))
            neg_vec  = self.model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
            query_vec = query_vec - (neg_vec * 0.8)

        candidate_size = len(self.df)
        semantic_hits  = util.semantic_search(query_vec, self.embeddings, top_k=candidate_size)

        # ── 2. SCORE LÉXICO (BM25) ────────────────────────────────────────────
        bm25_tokens = []
        for c in conceptos:
            # Tokens en idioma original (ya con variantes via normalize_text)
            bm25_tokens.extend(normalize_text(c))
            try:
                idiomas_target = ['en', 'es', 'pt']
                for lang in idiomas_target:
                    try:
                        traduccion = GoogleTranslator(source='auto', target=lang).translate(c)
                        if traduccion and traduccion.lower() != c.lower():
                             # normalize_text ya expande variantes
                            bm25_tokens.extend(normalize_text(traduccion))
                                
                            # NUEVO: también agregar variantes del texto crudo traducido
                            # Esto captura "make-up" → "makeup" directamente
                            for variant in expand_token_variants(traduccion.lower().strip()):
                                if len(variant) > 2:
                                    bm25_tokens.append(variant)
                    except:
                        continue
            except ImportError:
                pass
            
            # Compound token sin espacios
            if ' ' in c:
                bm25_tokens.append(c.replace(' ', '').lower())
                bm25_tokens.append(c.replace(' ', '-').lower())  # NUEVO

        tokenized_query = list(dict.fromkeys(bm25_tokens))  # dedup preservando orden
        bm25_scores = self.bm25.get_scores(tokenized_query)

        negative_keywords = set()
        if negative_query:
            neg_tokens = []
            for c in parsear_conceptos(negative_query):
                neg_tokens.extend(normalize_text(c))
                if ' ' in c:
                    neg_tokens.append(c.replace(' ', '').lower())
            negative_keywords = set(neg_tokens)

            if bm25_negative_penalty > 0:
                for idx in range(len(bm25_scores)):
                    if idx >= len(self.df):
                        continue
                    row   = self.df.iloc[idx]
                    texto = str(row.get('channel_title', '') or row.get('common_title', '')) + \
                            " " + str(row.get('channel_description', '') or row.get('desc_final', ''))
                    if any(kw in texto.lower() for kw in negative_keywords):
                        bm25_scores[idx] *= (1 - bm25_negative_penalty)

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0


        # ── 3. FUSIÓN ─────────────────────────────────────────────────────────
        score_components = {}

        for hit in semantic_hits[0]:
            idx = hit['corpus_id']
            if idx < len(self.df):
                score_components[idx] = {"semantic": float(hit['score']), "lexical": 0.0, "metric": 0.0}

        for idx, bm25_score in enumerate(bm25_scores):
            if idx >= len(self.df):
                continue
            normalized_bm25 = bm25_score / max_bm25
            if idx in score_components:
                score_components[idx]["lexical"] = normalized_bm25
            else:
                score_components[idx] = {"semantic": 0.0, "lexical": normalized_bm25, "metric": 0.0}

        # Métricas solo para YouTube
        if self.es_youtube and self.metric_stats:
            for idx in score_components:
                row = self.df.iloc[idx]
                score_components[idx]["metric"] = calcular_score_metricas(
                    row.to_dict(), self.metric_stats
                )

        combined_scores = {
            idx: (
                c["semantic"] * semantic_weight +
                c["lexical"]  * lexical_weight  +
                c["metric"]   * metric_weight
            )
            for idx, c in score_components.items()
        }

        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # ── 4. CONSTRUIR RESULTADOS ───────────────────────────────────────────
        results = []

        for idx, combined_score in sorted_indices:
            if len(results) >= top_k:
                break

            row = self.df.iloc[idx]

            # Filtro duro de negativos
            if hard_negative_filter and negative_keywords:
                texto_filtrar = " ".join([
                    str(row.get('channel_title', '')    or ''),
                    str(row.get('common_title', '')     or ''),
                    str(row.get('channel_description','') or ''),
                    str(row.get('desc_final', '')       or ''),
                    str(row.get('channel_keywords', '') or ''),
                ])
                if any(kw in normalize_text(texto_filtrar) for kw in negative_keywords):
                    continue

            # Filtros numéricos
            if filters:
                if 'score_min' in filters and float(row.get('score', 0) or 0) < filters['score_min']:
                    continue
                if 'genero' in filters and row.get('genero') != filters['genero']:
                    continue

            titulo = get_titulo(row, self.es_youtube)
            desc   = get_descripcion(row, self.es_youtube)
            comps  = score_components.get(idx, {})

            results.append({
                "score": float(combined_score),
                "score_breakdown": {
                    "semantico":  round(comps.get("semantic", 0) * semantic_weight, 4),
                    "lexico":     round(comps.get("lexical",  0) * lexical_weight,  4),
                    "metricas":   round(comps.get("metric",   0) * metric_weight,   4),
                    "raw_metric": round(comps.get("metric",   0), 4),
                },
                "titulo":      titulo,
                "descripcion": desc[:200] + ("..." if len(desc) > 200 else ""),
                "metadata":    row.to_dict(),
                "es_youtube":  self.es_youtube,
            })

        return results
import os
import time
import json
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

# Construimos detector solo con los idiomas que nos interesan
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
    if detected == Language.SPANISH: return "es"
    elif detected == Language.ENGLISH: return "en"
    elif detected == Language.PORTUGUESE: return "pt"
    return "es"

load_dotenv()

@lru_cache(maxsize=10000)
def normalize_text(text: str):
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

class RecommendationEngine:
    def __init__(self, model_name='intfloat/multilingual-e5-base'):
        """
        Motor de búsqueda híbrido.
        AHORA 100% CONECTADO A POSTGRES (Kidscorp Producto).
        """
        print(f"🧠 Cargando Modelo E5 (Solo para queries): {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.df = pd.DataFrame()
        self.embeddings = None
        self.bm25 = None
        self.source_name = ""

    def _load_from_db(self, fuente):
        """Descarga el BM25, los vectores E5 y la metadata desde Postgres"""
        print(f"\n☁️ Conectando a Postgres para cargar: '{fuente}'")
        start = time.time()
        
        conn = None
        try:
            # Nos conectamos SIEMPRE a la base de datos de Producto donde guardamos los vectores
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"), 
                database=os.getenv("DB_NAME"), # kidscorp_producto
                user=os.getenv("DB_USER"), 
                password=os.getenv("DB_PASS"),
                port="5432", 
                sslmode="require"
            )
            cursor = conn.cursor()
            
            # 1. Cargar BM25
            print("   📥 Descargando índice léxico (BM25)...")
            cursor.execute("SELECT archivo_pickle FROM keywordsearch.archivos_bm25 WHERE fuente = %s", (fuente,))
            row = cursor.fetchone()
            if row:
                self.bm25 = pickle.loads(row[0])
                print("   ✅ BM25 cargado y listo en memoria.")
            else:
                print(f"   ❌ ERROR: No se encontró BM25 para '{fuente}' en la DB.")
                
            # 2. Cargar Vectores y Metadata
            print("   📥 Descargando vectores semánticos (E5) y metadata...")
            # IMPORTANTE: ORDER BY id ASC para que coincidan perfectamente con el índice interno del BM25
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
                    # Convertimos el string de la DB "[0.1, 0.2...]" a una lista de Python
                    tensor_list.append(json.loads(emb_str))
                    
                # Reconstruimos el DataFrame y el Tensor en memoria
                self.df = pd.DataFrame(df_records)
                self.embeddings = torch.tensor(tensor_list)
                print(f"   ✅ {len(self.df)} registros cargados y sincronizados.")
            else:
                print(f"   ❌ ERROR: No se encontraron vectores para '{fuente}' en la DB.")
                
            cursor.close()
            
        except Exception as e:
            print(f"❌ Error de conexión a DB: {e}")
        finally:
            if conn: 
                conn.close()
                
        print(f"⏱️ Tiempo total de descarga: {time.time() - start:.2f}s\n")

    def load_from_postgres(self, db_config=None, limit=None, force_refresh=False):
        """Compatible con app.py para YouTube"""
        self.source_name = "youtube_channels_db"
        self._load_from_db(self.source_name)

    def load_from_json(self, json_path, force_refresh=False):
        """Compatible con app.py para las Apps (ya no lee el JSON, usa la DB)"""
        # Extraemos "mp.audience.2" de la ruta "apps_scraped_2024/mp.audience.2.json"
        fuente = os.path.basename(json_path).replace('.json', '')
        self.source_name = fuente
        self._load_from_db(fuente)

    def search(self, 
               query, 
               negative_query=None, 
               top_k=5, 
               filters=None,
               semantic_weight=0.7,
               lexical_weight=0.3,
               hard_negative_filter=True,
               bm25_negative_penalty=0.7,
               negative_boost_factor=1):
        """
        Búsqueda híbrida inalterada.
        """
        if self.embeddings is None or self.bm25 is None:
            print("❌ Embeddings o BM25 no están cargados en memoria")
            return []

        total_weight = semantic_weight + lexical_weight
        if abs(total_weight - 1.0) > 0.01:
            semantic_weight = semantic_weight / total_weight
            lexical_weight = lexical_weight / total_weight

        # ==========================================
        # BÚSQUEDA SEMÁNTICA (E5)
        # ==========================================
        query_text = "query: " + query
        query_vec = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        
        if negative_query:
            neg_text = "query: " + negative_query
            neg_vec = self.model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
            query_vec = query_vec - (neg_vec * 0.8)
        
        candidate_size = max(1000, top_k * 10, int(len(self.df) * 0.1))
        candidate_size = min(candidate_size, len(self.df))
        
        semantic_hits = util.semantic_search(query_vec, self.embeddings, top_k=candidate_size)
        
        # ==========================================
        # BÚSQUEDA LÉXICA (BM25)
        # ==========================================
        tokenized_query = normalize_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        negative_keywords = set()
        if negative_query:
            negative_keywords = set(normalize_text(negative_query))
            
            if bm25_negative_penalty > 0:
                for idx in range(len(bm25_scores)):
                    row = self.df.iloc[idx]
                    texto = (
                        str(row.get('channel_title', '')) + " " +
                        str(row.get('common_title', '')) + " " +
                        str(row.get('channel_description', '')) + " " +
                        str(row.get('desc_final', ''))
                    ).lower()
                    
                    if any(keyword in texto for keyword in negative_keywords):
                        bm25_scores[idx] *= (1 - bm25_negative_penalty)
        
        # ==========================================
        # FUSIÓN DE SCORES
        # ==========================================
        combined_scores = {}
        
        for hit in semantic_hits[0]:
            idx = hit['corpus_id']
            combined_scores[idx] = float(hit['score']) * semantic_weight
        
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for idx, bm25_score in enumerate(bm25_scores):
            normalized_bm25 = bm25_score / max_bm25
            
            if idx in combined_scores:
                combined_scores[idx] += normalized_bm25 * lexical_weight
            else:
                combined_scores[idx] = normalized_bm25 * lexical_weight
        
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # ==========================================
        # APLICAR FILTROS Y CONSTRUIR RESULTADOS
        # ==========================================
        results = []
        excluded_count = 0
        
        for idx, combined_score in sorted_indices:
            if len(results) >= top_k:
                break
            
            row = self.df.iloc[idx]
            
            if hard_negative_filter and negative_keywords:
                texto_para_filtrar = " ".join([
                    str(row.get('channel_title', '')),
                    str(row.get('common_title', '')),
                    str(row.get('channel_description', '')),
                    str(row.get('desc_final', '')),
                    str(row.get('channel_bs_ch_keywords', '')), 
                    str(row.get('genero', ''))                  
                ])
                
                texto_normalizado = normalize_text(texto_para_filtrar)
                has_negative = any(keyword in texto_normalizado for keyword in negative_keywords)

                if has_negative:
                    excluded_count += 1
                    continue
            
            if filters:
                if 'genero' in filters and filters['genero']:
                    if filters['genero'].lower() not in str(row.get('genero', '')).lower():
                        continue
                
                if 'score_min' in filters:
                    try:
                        val = float(row.get('score', 0))
                    except: 
                        val = 0
                    if val < filters['score_min']:
                        continue
                
                if 'must_contain' in filters and filters['must_contain']:
                    texto_completo = str(row.get('channel_title', '')) + " " + str(row.get('channel_description', ''))
                    if filters['must_contain'].lower() not in texto_completo.lower():
                        continue
                
                if 'must_not_contain' in filters and filters['must_not_contain']:
                    texto_completo = str(row.get('channel_title', '')) + " " + str(row.get('channel_description', ''))
                    if filters['must_not_contain'].lower() in texto_completo.lower():
                        continue
            
            title = row.get('channel_title') or row.get('common_title') or "Sin Título"
            desc = row.get('channel_description') or row.get('desc_final') or ""
            
            results.append({
                "score": float(combined_score),
                "titulo": title,
                "descripcion": desc[:200] + ("..." if len(desc) > 200 else ""),
                "metadata": row.to_dict()
            })
        
        return results
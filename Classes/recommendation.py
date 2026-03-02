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
def parsear_conceptos(query: str):
    """
    Separa la query por comas si existen, limpiando los espacios extra.
    Ej: "back to school, tennis, jockey" -> ["back to school", "tennis", "jockey"]
    """
    if ',' in query:
        return [concepto.strip() for concepto in query.split(',') if concepto.strip()]
    return [query.strip()]

class RecommendationEngine:
    def __init__(self, model_name='intfloat/multilingual-e5-base'):
        print(f"🧠 Cargando Modelo E5 (Solo para queries): {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.df = pd.DataFrame()
        self.embeddings = None
        self.bm25 = None
        self.source_name = ""

    def _load_from_db(self, fuente):
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
            
            # 1. Cargar BM25
            print("   📥 Descargando índice léxico (BM25)...")
            cursor.execute("SELECT archivo_pickle FROM keywordsearch.archivos_bm25 WHERE fuente = %s", (fuente,))
            row = cursor.fetchone()
            if row:
                self.bm25 = pickle.loads(row[0])
                print("   ✅ BM25 cargado y listo en memoria.")
            else:
                print(f"   ❌ ERROR: No se encontró BM25 para '{fuente}'.")
                
            # 2. Cargar Vectores y Metadata
            print("   📥 Descargando vectores semánticos (E5) y metadata...")
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
                
                print("   ⚙️ Procesando memoria (Esto puede tomar unos segundos en YouTube)...")
                for meta, emb_str in rows:
                    df_records.append(meta)
                    # OPTIMIZACIÓN CRÍTICA: np.fromstring no satura la memoria RAM como json.loads
                    arr = np.fromstring(emb_str[1:-1], sep=',')
                    tensor_list.append(arr)
                    
                self.df = pd.DataFrame(df_records)
                # Convertimos todo a un solo bloque de memoria contigua en Torch
                self.embeddings = torch.tensor(np.array(tensor_list), dtype=torch.float32)
                
                print(f"   ✅ {len(self.df)} registros cargados y sincronizados.")
            else:
                print(f"   ❌ ERROR: No se encontraron vectores para '{fuente}'.")
                
            cursor.close()
            
        except Exception as e:
            print(f"❌ Error de conexión a DB: {e}")
        finally:
            if conn: 
                conn.close()
                
        print(f"⏱️ Tiempo total de descarga: {time.time() - start:.2f}s\n")

    def load_from_postgres(self, db_config=None, limit=None, force_refresh=False):
        self.source_name = "youtube_channels_db"
        self._load_from_db(self.source_name)

    def load_from_json(self, json_path, force_refresh=False):
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
               bm25_negative_penalty=0.7):
        
        if self.embeddings is None or self.bm25 is None:
            return []

        total_weight = semantic_weight + lexical_weight
        if abs(total_weight - 1.0) > 0.01:
            semantic_weight = semantic_weight / total_weight
            lexical_weight = lexical_weight / total_weight

        conceptos_lista = parsear_conceptos(query)

        # 1. PARSEAMOS LOS CONCEPTOS
        conceptos = parsear_conceptos(query)

        # SEMÁNTICA (E5)
        # Pasamos los conceptos unidos por coma para que E5 entienda la enumeración
        query_text = "query: " + ", ".join(conceptos)
        query_vec = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        
        if negative_query:
            neg_conceptos = parsear_conceptos(negative_query)
            neg_text = "query: " + ", ".join(neg_conceptos)
            neg_vec = self.model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
            query_vec = query_vec - (neg_vec * 0.8)
        
        candidate_size = max(1000, top_k * 10, int(len(self.df) * 0.1))
        candidate_size = min(candidate_size, len(self.df))
        
        semantic_hits = util.semantic_search(query_vec, self.embeddings, top_k=candidate_size)
        
        # LÉXICA (BM25)
        bm25_tokens = []
        for c in conceptos:
            # Agregamos palabras sueltas ("back", "school", "tennis")
            bm25_tokens.extend(normalize_text(c)) 
            # Si hay espacios, agregamos también la palabra fusionada ("backtoschool")
            if ' ' in c:
                bm25_tokens.append(c.replace(' ', '').lower())
                
        # Eliminamos duplicados manteniendo el formato de lista
        tokenized_query = list(dict.fromkeys(bm25_tokens))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # NEGATIVOS LÉXICOS (BM25)
        negative_keywords = set()
        if negative_query:
            # Hacemos el mismo tratamiento (separar y fusionar) para las palabras a excluir
            neg_conceptos = parsear_conceptos(negative_query)
            neg_tokens = []
            for c in neg_conceptos:
                neg_tokens.extend(normalize_text(c))
                if ' ' in c:
                    neg_tokens.append(c.replace(' ', '').lower())
            
            negative_keywords = set(neg_tokens)
            
            if bm25_negative_penalty > 0:
                for idx in range(len(bm25_scores)):
                    # ESCUDO: Evitar IndexError si BM25 tiene más datos que la DB
                    if idx >= len(self.df): 
                        continue
                        
                    row = self.df.iloc[idx]
                    texto = str(row.get('channel_title', '')) + " " + str(row.get('channel_description', ''))
                    if any(keyword in texto.lower() for keyword in negative_keywords):
                        bm25_scores[idx] *= (1 - bm25_negative_penalty)
        
        # FUSIÓN DE SCORES
        combined_scores = {}
        
        for hit in semantic_hits[0]:
            idx = hit['corpus_id']
            # ESCUDO: Asegurar que el ID exista en el DataFrame
            if idx < len(self.df):
                combined_scores[idx] = float(hit['score']) * semantic_weight
        
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for idx, bm25_score in enumerate(bm25_scores):
            if idx >= len(self.df): 
                continue # ESCUDO
                
            normalized_bm25 = bm25_score / max_bm25
            if idx in combined_scores:
                combined_scores[idx] += normalized_bm25 * lexical_weight
            else:
                combined_scores[idx] = normalized_bm25 * lexical_weight
        
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # RESULTADOS
        results = []
        
        for idx, combined_score in sorted_indices:
            if len(results) >= top_k:
                break
                
            row = self.df.iloc[idx]
            
            if hard_negative_filter and negative_keywords:
                texto_para_filtrar = " ".join([
                    str(row.get('channel_title', '')),
                    str(row.get('channel_description', '')),
                    str(row.get('channel_bs_ch_keywords', ''))
                ])
                if any(keyword in normalize_text(texto_para_filtrar) for keyword in negative_keywords):
                    continue
            
            if filters:
                if 'score_min' in filters and float(row.get('score', 0)) < filters['score_min']:
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
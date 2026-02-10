import os
import time
import json
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import unicodedata

def limpiar_texto(texto):
    """Elimina tildes y convierte a minúsculas para comparaciones precisas."""
    if not texto: return ""
    texto = str(texto).lower()
    # Eliminar acentos
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

load_dotenv()

class RecommendationEngine:
    def __init__(self, 
                 model_name='intfloat/multilingual-e5-base',
                 cache_dir="cache"):
        """
        Motor de búsqueda híbrido usando E5 (semántico) + BM25 (léxico).
        
        COMPATIBLE CON TU INTERFAZ ACTUAL - Solo reemplaza el archivo en Classes/
        """
        print(f"🧠 Cargando Modelo E5: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.df = pd.DataFrame()
        self.embeddings = None
        self.bm25 = None
        self.source_name = ""
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def load_from_postgres(self, db_config=None, limit=10000):
        """Carga canales de YouTube desde PostgreSQL"""
        self.source_name = "youtube_channels_db_hybrid"
        print("🔌 Conectando a PostgreSQL...")
        
        conn = None
        try:
            host = db_config.get('host') if db_config else os.getenv("DB_HOST")
            database = db_config.get('database') if db_config else os.getenv("DB_NAME")
            user = db_config.get('user') if db_config else os.getenv("DB_USER")
            password = db_config.get('password') if db_config else os.getenv("DB_PASS")

            conn = psycopg2.connect(
                host=host, database=database, user=user, password=password,
                port="5432", sslmode="require"
            )
            
            query = f"""
                SELECT channel_title, channel_description, channel_customurl, channel_bs_ch_keywords 
                FROM ods.tbl_canales 
                LIMIT {limit};
            """
            
            self.df = pd.read_sql_query(query, conn)
            self.df.fillna('', inplace=True)
            
            # Texto para E5 (con prefijo 'passage:')
            self.df['texto_ia'] = "passage: " + (
                "Canal: " + self.df['channel_title'] + ". " +
                "Descripción: " + self.df['channel_description'] + ". " +
                "Keywords: " + self.df['channel_bs_ch_keywords']
            )
            
            # Texto para BM25 (sin prefijo, limpio)
            self.df['texto_bm25'] = (
                self.df['channel_title'] + " " +
                self.df['channel_description'] + " " +
                self.df['channel_bs_ch_keywords']
            )
            
            print(f"✅ {len(self.df)} canales cargados desde DB.")
            self._manage_embeddings()
            
        except Exception as e:
            print(f"❌ Error SQL: {e}")
        finally:
            if conn: 
                conn.close()

    def load_from_json(self, json_path):
        """Carga apps desde archivo JSON"""
        self.source_name = os.path.basename(json_path).replace('.json', '') + "_hybrid"
        print(f"📂 Cargando JSON: {json_path}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.df = pd.DataFrame(data)
            self.df.fillna('', inplace=True)
            
            self.df['desc_final'] = self.df.apply(
                lambda row: row.get('desc_larga', '') if len(str(row.get('desc_larga', ''))) > 10 else row.get('desc_corta', ''), 
                axis=1
            )
            
            col_titulo = 'titulo_store' if 'titulo_store' in self.df.columns else 'titulo'
            self.df['common_title'] = self.df[col_titulo]
            
            # Texto para E5 (con prefijo 'passage:')
            self.df['texto_ia'] = "passage: " + (
                "App: " + self.df['common_title'].astype(str) + ". " +
                "Género: " + self.df.get('genero', '').astype(str) + ". " +
                "Descripción: " + self.df['desc_final'].astype(str)
            )
            
            # Texto para BM25 (sin prefijo, limpio)
            self.df['texto_bm25'] = (
                self.df['common_title'].astype(str) + " " +
                self.df.get('genero', '').astype(str) + " " +
                self.df['desc_final'].astype(str)
            )
            
            self.df = self.df[self.df['texto_ia'].str.len() > 25].reset_index(drop=True)
            
            print(f"✅ {len(self.df)} apps cargadas.")
            self._manage_embeddings()
            
        except Exception as e:
            print(f"❌ Error JSON: {e}")

    def _manage_embeddings(self):
        """Genera o carga embeddings E5 + índice BM25"""
        cache_file = os.path.join(self.cache_dir, f"{self.source_name}_embeddings.pkl")
        bm25_cache_file = os.path.join(self.cache_dir, f"{self.source_name}_bm25.pkl")
        
        # === EMBEDDINGS E5 ===
        if os.path.exists(cache_file):
            print(f"💾 Cargando caché E5: {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if len(cached_data) == len(self.df):
                self.embeddings = cached_data
                print("✅ Embeddings E5 listos.")
            else:
                print("⚠️ Caché inválido, regenerando...")
                self._generate_embeddings(cache_file)
        else:
            self._generate_embeddings(cache_file)
        
        # === ÍNDICE BM25 ===
        if os.path.exists(bm25_cache_file):
            print(f"💾 Cargando caché BM25: {bm25_cache_file}...")
            with open(bm25_cache_file, 'rb') as f:
                self.bm25 = pickle.load(f)
            print("✅ Índice BM25 listo.")
        else:
            self._generate_bm25_index(bm25_cache_file)

    def _generate_embeddings(self, cache_file):
        """Genera embeddings E5"""
        print("⚡ Generando embeddings E5...")
        start = time.time()
        self.embeddings = self.model.encode(
            self.df['texto_ia'].tolist(), 
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"⏱️ Tiempo E5: {time.time() - start:.2f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"💾 Caché guardado: {cache_file}")

    def _generate_bm25_index(self, cache_file):
        """Genera índice BM25"""
        print("🔤 Creando índice BM25...")
        start = time.time()
        
        # Tokenizar corpus (simple split por espacios)
        tokenized_corpus = [doc.lower().split() for doc in self.df['texto_bm25'].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"⏱️ Tiempo BM25: {time.time() - start:.2f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"💾 Caché guardado: {cache_file}")

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
        Búsqueda híbrida compatible con tu interfaz Streamlit.
        
        MISMA FIRMA QUE TU CÓDIGO ORIGINAL - No necesitas cambiar app.py
        Los parámetros nuevos son opcionales con defaults inteligentes.
        
        Args:
            query: Consulta de búsqueda
            negative_query: Términos a penalizar/excluir (opcional)
            top_k: Número de resultados
            filters: Dict con 'genero', 'score_min', etc.
            semantic_weight: Peso búsqueda semántica (0-1)
            lexical_weight: Peso búsqueda léxica (0-1)
            hard_negative_filter: Si True, EXCLUYE resultados con negative keywords (recomendado)
            bm25_negative_penalty: Penalización BM25 para negative keywords (0-1, 0.7 = -70%)
            negative_boost_factor: Boost para resultados SIN negative keywords (>1.0)
        
        Returns:
            Lista de resultados con estructura:
            {
                'score': float,
                'titulo': str,
                'descripcion': str,
                'metadata': dict
            }
        """
        if self.embeddings is None or self.bm25 is None:
            print("❌ Embeddings o BM25 no inicializados")
            return []

        # Normalizar pesos
        total_weight = semantic_weight + lexical_weight
        if abs(total_weight - 1.0) > 0.01:
            semantic_weight = semantic_weight / total_weight
            lexical_weight = lexical_weight / total_weight

        # ==========================================
        # BÚSQUEDA SEMÁNTICA (E5)
        # ==========================================
        query_text = "query: " + query
        query_vec = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        
        # Query negativa
        if negative_query:
            neg_text = "query: " + negative_query
            neg_vec = self.model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
            query_vec = query_vec - (neg_vec * 0.8)
        
        # Búsqueda semántica
        semantic_hits = util.semantic_search(query_vec, self.embeddings, top_k=min(len(self.df), top_k * 20))
        
        # ==========================================
        # BÚSQUEDA LÉXICA (BM25)
        # ==========================================
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Preparar negative keywords para filtrado/penalización
        negative_keywords = set()
        if negative_query:
            negative_keywords = set(negative_query.lower().split())
            
            # Penalizar scores BM25 si contienen negative keywords
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
        
        # Agregar scores semánticos
        for hit in semantic_hits[0]:
            idx = hit['corpus_id']
            combined_scores[idx] = float(hit['score']) * semantic_weight
        
        # Normalizar y agregar scores BM25, REVISAR!!!
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for idx, bm25_score in enumerate(bm25_scores):
            normalized_bm25 = bm25_score / max_bm25
            
            if idx in combined_scores:
                combined_scores[idx] += normalized_bm25 * lexical_weight
            else:
                combined_scores[idx] = normalized_bm25 * lexical_weight
        
        # Ordenar por score combinado
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # ==========================================
        # APLICAR FILTROS Y CONSTRUIR RESULTADOS
        # ==========================================
        results = []
        excluded_count = 0
        boosted_count = 0
        
        for idx, combined_score in sorted_indices:
            if len(results) >= top_k:
                break
            
            row = self.df.iloc[idx]
            
            # === FILTRO DURO DE NEGATIVE KEYWORDS ===
            if hard_negative_filter and negative_keywords:
                # 1. Normalizamos las palabras negativas una sola vez (puedes sacarlo del loop por eficiencia)
                neg_keywords_clean = [limpiar_texto(k) for k in negative_keywords]
                
                # 2. Construimos un texto que incluya ABSOLUTAMENTE TODO
                texto_para_filtrar = " ".join([
                    str(row.get('channel_title', '')),
                    str(row.get('common_title', '')),
                    str(row.get('channel_description', '')),
                    str(row.get('desc_final', '')),
                    str(row.get('channel_bs_ch_keywords', '')), 
                    str(row.get('genero', ''))                  
                ])
                
                texto_limpio = limpiar_texto(texto_para_filtrar)
                
                # 3. Buscamos coincidencias
                has_negative = any(keyword in texto_limpio for keyword in neg_keywords_clean)
                
                if has_negative:
                    excluded_count += 1
                    continue  # EXCLUIR COMPLETAMENTE
            
            # === FILTROS DUROS ===
            if filters:
                # Filtro por género
                if 'genero' in filters and filters['genero']:
                    if filters['genero'].lower() not in str(row.get('genero', '')).lower():
                        continue
                
                # Filtro por score mínimo
                if 'score_min' in filters:
                    try:
                        val = float(row.get('score', 0))
                    except: 
                        val = 0
                    if val < filters['score_min']:
                        continue
                
                # Filtro por keywords que DEBEN aparecer
                if 'must_contain' in filters and filters['must_contain']:
                    texto_completo = str(row.get('channel_title', '')) + " " + str(row.get('channel_description', ''))
                    if filters['must_contain'].lower() not in texto_completo.lower():
                        continue
                
                # Filtro por keywords que NO deben aparecer
                if 'must_not_contain' in filters and filters['must_not_contain']:
                    texto_completo = str(row.get('channel_title', '')) + " " + str(row.get('channel_description', ''))
                    if filters['must_not_contain'].lower() in texto_completo.lower():
                        continue
            
            # === EXTRAER DATOS (Compatible con tu interfaz) ===
            title = row.get('channel_title') or row.get('common_title') or "Sin Título"
            desc = row.get('channel_description') or row.get('desc_final') or ""
            
            results.append({
                "score": float(combined_score),
                "titulo": title,
                "descripcion": desc[:200] + ("..." if len(desc) > 200 else ""),
                "metadata": row.to_dict()
            })
        
        return results
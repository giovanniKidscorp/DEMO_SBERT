import os
import time
import json
import pickle
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder # <--- Nuevo Import
from dotenv import load_dotenv
import numpy as np
# Cargar variables de entorno
load_dotenv()

class RecommendationEngine:
    def __init__(self, 
                 bi_model_name='paraphrase-multilingual-MiniLM-L12-v2', 
                 cross_model_name='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', # <--- Modelo Juez
                 cache_dir="cache"):
        """
        Inicializa el motor híbrido (Buscador Rápido + Juez Preciso).
        """
        print(f"🧠 Cargando Bi-Encoder (Buscador): {bi_model_name}...")
        self.bi_encoder = SentenceTransformer(bi_model_name)
        
        print(f"⚖️ Cargando Cross-Encoder (Juez): {cross_model_name}...")
        self.cross_encoder = CrossEncoder(cross_model_name)
        
        self.df = pd.DataFrame()
        self.embeddings = None
        self.source_name = ""
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def load_from_postgres(self, db_config=None, limit=10000):
        self.source_name = "youtube_channels_db"
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
            
            # Traemos customurl y keywords para el contexto
            query = f"""
                SELECT channel_title, channel_description, channel_customurl, channel_bs_ch_keywords 
                FROM ods.tbl_canales 
                LIMIT {limit};
            """
            
            self.df = pd.read_sql_query(query, conn)
            self.df.fillna('', inplace=True)
            
            # Texto rico para la IA (Bi-Encoder)
            self.df['texto_ia'] = (
                "Canal: " + self.df['channel_title'] + ". " +
                "Descripción: " + self.df['channel_description'] + ". " +
                "Keywords: " + self.df['channel_bs_ch_keywords']
            )
            
            print(f"✅ {len(self.df)} canales cargados desde DB.")
            self._manage_embeddings()
            
        except Exception as e:
            print(f"❌ Error SQL: {e}")
        finally:
            if conn: conn.close()

    def load_from_json(self, json_path):
        self.source_name = os.path.basename(json_path).replace('.json', '')
        print(f"📂 Cargando JSON: {json_path}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.df = pd.DataFrame(data)
            self.df.fillna('', inplace=True)
            
            # Lógica de descripción
            self.df['desc_final'] = self.df.apply(
                lambda row: row.get('desc_larga', '') if len(str(row.get('desc_larga', ''))) > 10 else row.get('desc_corta', ''), 
                axis=1
            )
            
            col_titulo = 'titulo_store' if 'titulo_store' in self.df.columns else 'titulo'
            self.df['common_title'] = self.df[col_titulo]
            
            # Texto rico para la IA
            self.df['texto_ia'] = (
                "App: " + self.df['common_title'].astype(str) + ". " +
                "Género: " + self.df.get('genero', '').astype(str) + ". " +
                "Descripción: " + self.df['desc_final'].astype(str)
            )
            
            self.df = self.df[self.df['texto_ia'].str.len() > 20].reset_index(drop=True)
            
            print(f"✅ {len(self.df)} apps cargadas.")
            self._manage_embeddings()
            
        except Exception as e:
            print(f"❌ Error JSON: {e}")

    def _manage_embeddings(self):
        """Gestiona la caché de vectores del Bi-Encoder"""
        cache_file = os.path.join(self.cache_dir, f"{self.source_name}_embeddings.pkl")
        
        if os.path.exists(cache_file):
            print(f"💾 Cargando caché: {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if len(cached_data) == len(self.df):
                self.embeddings = cached_data
                print("✅ Embeddings listos.")
                return

        print("⚡ Generando nuevos embeddings (Bi-Encoder)...")
        start = time.time()
        self.embeddings = self.bi_encoder.encode(
            self.df['texto_ia'].tolist(), 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print(f"⏱️ Tiempo: {time.time() - start:.2f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def search(self, query, negative_query=None, top_k=5, filters=None):
        """
        Búsqueda en dos etapas:
        1. Retrieval (Bi-Encoder): Busca muchos candidatos (Rápido).
        2. Re-Ranking (Cross-Encoder): Re-ordena con precisión (Lento pero exacto).
        """
        if self.embeddings is None:
            return []

        # --- ETAPA 1: RETRIEVAL (La red amplia) ---
        # Pedimos más candidatos de los necesarios (ej: 50) para que el Re-Ranker tenga de donde elegir
        
        # 1.1 Vectorizar Query
        query_vec = self.bi_encoder.encode(query, convert_to_tensor=True)
        
        # 1.2 Lógica Negativa (Solo aplica al Bi-Encoder)
        if negative_query:
            neg_vec = self.bi_encoder.encode(negative_query, convert_to_tensor=True)
            query_vec = query_vec - (neg_vec * 0.5)

        # 1.3 Búsqueda Vectorial
        hits = util.semantic_search(query_vec, self.embeddings, top_k)
        
        # --- ETAPA 2: FILTRADO Y PREPARACIÓN PARA RE-RANKER ---
        pares_para_reranker = [] # Lista de pares [Query, Texto]
        indices_validos = []     # Índices originales en el DataFrame
        
        for hit in hits[0]:
            idx = hit['corpus_id']
            row = self.df.iloc[idx]
            
            # Aplicamos Hard Filters ANTES del Re-Ranker para no gastar CPU en lo que no sirve
            if filters:
                if 'genero' in filters and filters['genero']:
                    if filters['genero'].lower() not in str(row.get('genero', '')).lower():
                        continue
                if 'score_min' in filters:
                    try:
                        val = float(row.get('score', 0))
                    except: val = 0
                    if val < filters['score_min']:
                        continue

            # Si pasó los filtros, lo preparamos para el juez
            texto_documento = row['texto_ia']
            pares_para_reranker.append([query, texto_documento])
            indices_validos.append(idx)

        # Si no quedó nada después de filtrar, retornamos vacío
        if not pares_para_reranker:
            return []

        # --- ETAPA 3: RE-RANKING (El Juez) ---
        # El Cross-Encoder predice un score para cada par (Query, Documento)
        # Esto soluciona lo de "AS" vs "Asbri" porque analiza la relación real
        cross_scores = self.cross_encoder.predict(pares_para_reranker)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        cross_scores_probs = sigmoid(cross_scores)

        # --- ETAPA 4: ORDENAMIENTO FINAL ---
        resultados_finales = []
        for i, score in enumerate(cross_scores_probs):
            idx_original = indices_validos[i]
            row = self.df.iloc[idx_original]
            
            # Usamos el score del Cross-Encoder, que es más confiable
            # Normalmente viene en logits, a veces conviene normalizarlo con sigmoid, 
            # pero para ordenar sirve tal cual.
            
            title = row.get('channel_title') or row.get('common_title') or "Sin Título"
            desc = row.get('channel_description') or row.get('desc_final') or ""
            
            resultados_finales.append({
                "score": float(score), # Score del Cross-Encoder
                "titulo": title,
                "descripcion": desc[:200] + "...",
                "metadata": row.to_dict()
            })

        # Ordenamos de mayor a menor según el NUEVO score
        resultados_finales = sorted(resultados_finales, key=lambda x: x['score'], reverse=True)
        
        # Devolvemos solo el top_k solicitado
        return resultados_finales[:top_k]
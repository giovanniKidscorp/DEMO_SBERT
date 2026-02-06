import os
import time
import json
import pickle
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

class RecommendationEngine:
    def __init__(self, 
                 model_name='intfloat/multilingual-e5-base', # <--- MODELO NUEVO (Más listo)
                 cache_dir="cache"):
        """
        Motor de búsqueda optimizado usando E5 (Instruct Model).
        Es rápido (Bi-Encoder) pero preciso gracias al entrenamiento con instrucciones.
        """
        print(f"🧠 Cargando Modelo E5: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.df = pd.DataFrame()
        self.embeddings = None
        self.source_name = ""
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def load_from_postgres(self, db_config=None, limit=10000):
        self.source_name = "youtube_channels_db_e5" # Cambiamos nombre para no mezclar cache viejo
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
            
            # --- TRUCO DE E5: Prefijo 'passage: ' ---
            # Esto le dice al modelo que esto es contenido para ser buscado
            self.df['texto_ia'] = "passage: " + (
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
        self.source_name = os.path.basename(json_path).replace('.json', '') + "_e5"
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
            
            # --- TRUCO DE E5: Prefijo 'passage: ' ---
            self.df['texto_ia'] = "passage: " + (
                "App: " + self.df['common_title'].astype(str) + ". " +
                "Género: " + self.df.get('genero', '').astype(str) + ". " +
                "Descripción: " + self.df['desc_final'].astype(str)
            )
            
            self.df = self.df[self.df['texto_ia'].str.len() > 25].reset_index(drop=True) # >25 por el prefijo
            
            print(f"✅ {len(self.df)} apps cargadas.")
            self._manage_embeddings()
            
        except Exception as e:
            print(f"❌ Error JSON: {e}")

    def _manage_embeddings(self):
        cache_file = os.path.join(self.cache_dir, f"{self.source_name}_embeddings.pkl")
        
        if os.path.exists(cache_file):
            print(f"💾 Cargando caché: {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if len(cached_data) == len(self.df):
                self.embeddings = cached_data
                print("✅ Embeddings listos.")
                return

        print("⚡ Generando nuevos embeddings con E5 (Bi-Encoder)...")
        start = time.time()
        self.embeddings = self.model.encode(
            self.df['texto_ia'].tolist(), 
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True # E5 funciona mejor con vectores normalizados
        )
        print(f"⏱️ Tiempo: {time.time() - start:.2f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def search(self, query, negative_query=None, top_k=5, filters=None):
        if self.embeddings is None:
            return []

        # --- TRUCO DE E5: Prefijo 'query: ' ---
        # Esto separa semánticamente la búsqueda del documento
        query_text = "query: " + query
        query_vec = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        
        # Lógica Negativa
        if negative_query:
            neg_text = "query: " + negative_query
            neg_vec = self.model.encode(neg_text, convert_to_tensor=True, normalize_embeddings=True)
            query_vec = query_vec - (neg_vec * 0.4) # Factor suave

        # Búsqueda Rápida (Coseno)
        # Pedimos un poco más para tener margen de filtrado
        hits = util.semantic_search(query_vec, self.embeddings, top_k=top_k * 5)
        
        results = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            row = self.df.iloc[idx]
            score = hit['score']
            
            # FILTROS DUROS
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
            
            title = row.get('channel_title') or row.get('common_title') or "Sin Título"
            desc = row.get('channel_description') or row.get('desc_final') or ""
            
            results.append({
                "score": float(score),
                "titulo": title,
                "descripcion": desc[:200] + "...",
                "metadata": row.to_dict()
            })
            
            if len(results) >= top_k:
                break
        
        return results
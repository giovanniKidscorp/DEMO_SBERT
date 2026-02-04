import os
import time
import json
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Cargar variables de entorno si existen (.env)
load_dotenv()

class RecommendationEngine:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', cache_dir="cache"):
        """
        Inicializa el motor de recomendación.
        :param model_name: Nombre del modelo de HuggingFace.
        :param cache_dir: Carpeta donde se guardarán los archivos .pkl de embeddings.
        """
        print(f"🧠 Cargando modelo IA: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.df = pd.DataFrame() # DataFrame activo
        self.embeddings = None   # Tensores activos
        self.source_name = ""    # Nombre para identificar el caché
        
        # Crear carpeta de caché si no existe
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def load_from_postgres(self, db_config=None, limit=10000):
        """
        Carga datos de canales desde AWS RDS PostgreSQL.
        """
        self.source_name = "youtube_channels_db"
        print("🔌 Conectando a PostgreSQL...")
        
        conn = None
        try:
            # Usa config pasada por argumento o busca en variables de entorno
            host = db_config.get('host') if db_config else os.getenv("DB_HOST")
            database = db_config.get('database') if db_config else os.getenv("DB_NAME")
            user = db_config.get('user') if db_config else os.getenv("DB_USER")
            password = db_config.get('password') if db_config else os.getenv("DB_PASS")

            conn = psycopg2.connect(
                host=host, database=database, user=user, password=password,
                port="5432", sslmode="require"
            )
            
            query = f"""
                SELECT channel_title, channel_description, channel_bs_ch_keywords 
                FROM ods.tbl_canales 
                LIMIT {limit};
            """
            
            self.df = pd.read_sql_query(query, conn)
            self.df.fillna('', inplace=True)
            
            # Crear Texto Rico (Feature Engineering)
            self.df['texto_ia'] = (
                "Canal: " + self.df['channel_title'] + ". " +
                "Descripción: " + self.df['channel_description'] + ". " +
                "Keywords: " + self.df['channel_bs_ch_keywords']
            )
            
            print(f"✅ {len(self.df)} canales cargados desde DB.")
            self._manage_embeddings() # Calcular o Cargar Vectores
            
        except Exception as e:
            print(f"❌ Error SQL: {e}")
        finally:
            if conn: conn.close()

    def load_from_json(self, json_path):
        """
        Carga datos de Apps desde un archivo JSON (scraped data).
        """
        self.source_name = os.path.basename(json_path).replace('.json', '')
        print(f"📂 Cargando JSON: {json_path}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.df = pd.DataFrame(data)
            self.df.fillna('', inplace=True)
            
            # Lógica específica para Apps (basada en tu notebook)
            # Prioriza descripción larga, si es muy corta usa la resumen
            self.df['desc_final'] = self.df.apply(
                lambda row: row.get('desc_larga', '') if len(str(row.get('desc_larga', ''))) > 10 else row.get('desc_corta', ''), 
                axis=1
            )
            
            # Normalización de nombres de columnas para que el buscador funcione genérico
            # Si el json tiene 'titulo_store', lo copiamos a una columna común 'title'
            col_titulo = 'titulo_store' if 'titulo_store' in self.df.columns else 'titulo'
            self.df['common_title'] = self.df[col_titulo]
            
            self.df['texto_ia'] = (
                "App: " + self.df['common_title'].astype(str) + ". " +
                "Género: " + self.df.get('genero', '').astype(str) + ". " +
                "Descripción: " + self.df['desc_final'].astype(str)
            )
            
            # Filtro básico de calidad (eliminar vacíos)
            self.df = self.df[self.df['texto_ia'].str.len() > 20].reset_index(drop=True)
            
            print(f"✅ {len(self.df)} apps cargadas.")
            self._manage_embeddings()
            
        except Exception as e:
            print(f"❌ Error JSON: {e}")

    def _manage_embeddings(self):
        """
        Maneja la caché de embeddings para no gastar CPU innecesariamente.
        """
        cache_file = os.path.join(self.cache_dir, f"{self.source_name}_embeddings.pkl")
        
        # 1. Intentar cargar
        if os.path.exists(cache_file):
            print(f"💾 Cargando caché: {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Verificación simple de integridad (mismo tamaño)
            if len(cached_data) == len(self.df):
                self.embeddings = cached_data
                print("✅ Embeddings listos.")
                return

        # 2. Generar si no existe
        print("⚡ Generando nuevos embeddings (esto puede tardar)...")
        start = time.time()
        self.embeddings = self.model.encode(
            self.df['texto_ia'].tolist(), 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print(f"⏱️ Tiempo: {time.time() - start:.2f}s")
        
        # 3. Guardar
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print("💾 Embeddings guardados en disco.")

    def search(self, query, negative_query=None, top_k=5, filters=None):
        """
        Realiza la búsqueda semántica.
        :param query: Texto positivo (lo que buscas).
        :param negative_query: Texto negativo (lo que quieres evitar).
        :param top_k: Cantidad de resultados.
        :param filters: Dict con filtros duros {'genero': 'Action', 'score_min': 4.0}.
        """
        if self.embeddings is None:
            print("⚠️ No hay datos cargados. Usa load_from_postgres o load_from_json primero.")
            return []

        print(f"\n🔎 Buscando: '{query}' {f'(⛔ NO: {negative_query})' if negative_query else ''}")
        
        # 1. Vectorizar Query Positiva
        query_vec = self.model.encode(query, convert_to_tensor=True)
        
        # 2. Aritmética Vectorial (Negativa)
        if negative_query:
            neg_vec = self.model.encode(negative_query, convert_to_tensor=True)
            # Restamos el vector negativo (el 0.5 es un factor de suavizado)
            query_vec = query_vec - (neg_vec * 0.5)

        # 3. Búsqueda Semántica
        hits = util.semantic_search(query_vec, self.embeddings, top_k=top_k * 2) # Pedimos el doble para filtrar después
        
        results = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            row = self.df.iloc[idx]
            score = hit['score']
            
            # --- FILTROS DUROS (Hard Filters) ---
            if filters:
                # Ejemplo filtro género
                if 'genero' in filters and filters['genero']:
                    if filters['genero'].lower() not in str(row.get('genero', '')).lower():
                        continue
                # Ejemplo filtro score
                if 'score_min' in filters:
                    if float(row.get('score', 0) or 0) < filters['score_min']:
                        continue

            # Selección de título según el origen (DB o JSON)
            title = row.get('channel_title') or row.get('common_title') or "Sin Título"
            desc = row.get('channel_description') or row.get('desc_final') or ""
            
            results.append({
                "score": round(score, 3),
                "titulo": title,
                "descripcion": desc[:150] + "...", # Preview
                "metadata": row.to_dict() # Guardamos toda la info por si acaso
            })
            
            if len(results) >= top_k:
                break
                
        return results
import os
import pickle
import warnings
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch, Json
from dotenv import load_dotenv
from Classes.recommendation import RecommendationEngine

# Silenciar los warnings de Pandas sobre SQLAlchemy
warnings.filterwarnings('ignore')

load_dotenv()

# ==========================================
# 1. CREAR LAS DOS CONEXIONES (EL PUENTE)
# ==========================================
# Conexión 1: Para ESCRIBIR (Producto)
conn_prod = psycopg2.connect(
    host=os.getenv("DB_HOST"), 
    database=os.getenv("DB_NAME"), 
    user=os.getenv("DB_USER"), 
    password=os.getenv("DB_PASS"),
    port="5432", sslmode="require"
)

# Conexión 2: Para LEER (YouTube)
conn_yt = psycopg2.connect(
    host=os.getenv("DB_HOST"), 
    database=os.getenv("DB_NAME_YT"), # <--- ¡Fijate que usa la variable nueva!
    user=os.getenv("DB_USER"), 
    password=os.getenv("DB_PASS"),
    port="5432", sslmode="require"
)

def migrar_todo():
    cursor_prod = conn_prod.cursor()
    engine = RecommendationEngine(cache_dir="cache")
    
    # === PASO 1: BM25 (Ya lo hiciste, lo salteamos si quieres, o lo dejamos por seguridad) ===
    print("📦 PASO 1: SUBIENDO ÍNDICES LÉXICOS (BM25) ...")
    for f_name in os.listdir('cache'):
        if f_name.endswith('bm25.pkl'):
            fuente = f_name.replace('_hybrid_bm25.pkl', '')
            with open(os.path.join('cache', f_name), 'rb') as f:
                cursor_prod.execute("""
                    INSERT INTO keywordsearch.archivos_bm25 (fuente, archivo_pickle)
                    VALUES (%s, %s)
                    ON CONFLICT (fuente) DO UPDATE SET archivo_pickle = EXCLUDED.archivo_pickle;
                """, (fuente, psycopg2.Binary(f.read())))
    conn_prod.commit()
    print("✅ ¡Todos los BM25 subidos a Postgres!\n")

    def insertar_lote(fuente_nombre, df, tensores):
        if df.empty or tensores is None: return
        lista_vectores = tensores.tolist()
        datos_para_insertar = []
        
        for i, row in df.iterrows():
            item_id = str(row.get('app_id') or row.get('id') or row.get('channel_customurl') or i)
            metadata = row.to_dict()
            texto_bm25 = str(row.get('texto_bm25', ''))
            vector_str = "[" + ",".join(map(str, lista_vectores[i])) + "]"
            datos_para_insertar.append((fuente_nombre, item_id, texto_bm25, Json(metadata), vector_str))
        
        query = """
            INSERT INTO keywordsearch.vectores_e5 (fuente, item_id, texto_bm25, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """
        execute_batch(cursor_prod, query, datos_para_insertar, page_size=1000)
        conn_prod.commit()
        print(f"   ✅ {len(datos_para_insertar)} vectores insertados para '{fuente_nombre}'.")

    # === PASO 2: APPS (De archivos locales a Producto) ===
    print("🧬 PASO 2: SUBIENDO VECTORES DE APPS...")
    carpeta_apps = "apps_scraped_2024"
    if os.path.exists(carpeta_apps):
        for archivo in os.listdir(carpeta_apps):
            if archivo.endswith('.json'):
                ruta = os.path.join(carpeta_apps, archivo)
                fuente = archivo.replace('.json', '')
                print(f"⚙️ Procesando {fuente}...")
                engine.load_from_json(ruta)
                insertar_lote(fuente, engine.df, engine.embeddings)

    # === PASO 3: YOUTUBE (El puente mágico) ===
    print("\n⚙️ Procesando YouTube (Modo Seguro con 2 conexiones)...")
    yt_cache_file = 'cache/youtube_channels_db_hybrid_embeddings.pkl'
    
    if os.path.exists(yt_cache_file):
        with open(yt_cache_file, 'rb') as f:
            yt_embeddings = pickle.load(f)
            
        total_vectores = len(yt_embeddings)
        print(f"   📊 Encontramos {total_vectores} vectores en el caché.")
        
        # ACA ESTÁ LA MAGIA: Usamos conn_yt para leer la tabla
        df_yt = pd.read_sql_query(f"""
            SELECT channel_title, channel_description, channel_customurl, channel_bs_ch_keywords 
            FROM ods.tbl_canales 
            LIMIT {total_vectores};
        """, conn_yt) 
        df_yt.fillna('', inplace=True)
        
        df_yt['texto_bm25'] = (
            df_yt['channel_title'] + " " +
            df_yt['channel_description'] + " " +
            df_yt['channel_bs_ch_keywords']
        )
        
        # Y acá insertamos usando la función que guarda en conn_prod
        insertar_lote("youtube_channels_db", df_yt, yt_embeddings)
    else:
        print("   ❌ No se encontró el archivo caché de YouTube.")

    cursor_prod.close()
    conn_prod.close()
    conn_yt.close()
    print("\n🚀 ¡MIGRACIÓN COMPLETADA CON ÉXITO!")

if __name__ == "__main__":
    migrar_todo()
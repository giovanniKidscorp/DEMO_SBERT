import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Conexión a Kidscorp Producto
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"), password=os.getenv("DB_PASS"),
    port="5432", sslmode="require"
)
cursor = conn.cursor()

print("📦 SUBIENDO ÍNDICES LÉXICOS (BM25)...")
for f_name in os.listdir('cache'):
    if f_name.endswith('bm25.pkl'):
        fuente = f_name.replace('_hybrid_bm25.pkl', '')
        print(f"   ⬆️ Subiendo archivo: {fuente}...")
        
        with open(os.path.join('cache', f_name), 'rb') as f:
            cursor.execute("""
                INSERT INTO keywordsearch.archivos_bm25 (fuente, archivo_pickle)
                VALUES (%s, %s)
                ON CONFLICT (fuente) DO UPDATE SET archivo_pickle = EXCLUDED.archivo_pickle;
            """, (fuente, psycopg2.Binary(f.read())))
        
        # Guardamos en la base de datos INMEDIATAMENTE después de cada archivo
        conn.commit()
        print(f"   ✅ {fuente} guardado con éxito.")

cursor.close()
conn.close()
print("🚀 ¡Todos los BM25 están a salvo en Postgres!")
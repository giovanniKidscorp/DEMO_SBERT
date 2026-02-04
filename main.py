from Classes.recommendation import RecommendationEngine

# 1. Inicializar el motor (Carga el modelo una sola vez)
rec_sys = RecommendationEngine()

# --- ESCENARIO A: Búsqueda de Canales (PostgreSQL) ---
# Se conectará a tu DB, bajará los datos y creará el .pkl si no existe
rec_sys.load_from_postgres(limit=5000) 

# Buscar
resultados = rec_sys.search("videos de manualidades", negative_query="gameplay")
for r in resultados:
    print(f"📺 {r['score']} | {r['titulo']}")


# --- ESCENARIO B: Búsqueda de Apps (JSON) ---
# Cambiamos el contexto a Apps (el sistema cambia el DataFrame y Embeddings internos)
rec_sys.load_from_json("apps_scraped_2024/mp.audience.2.json")

# Buscar con filtros
filtros = {'genero': 'Education', 'score_min': 4.0}
resultados_apps = rec_sys.search("aprender matematicas", filters=filtros)

for r in resultados_apps:
    print(f"📱 {r['score']} | {r['titulo']} ({r['metadata'].get('genero')})")
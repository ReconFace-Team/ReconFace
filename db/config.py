# config.py
import os

# Rutas de almacenamiento (ajustadas para ../db/database.py)
BASE_EMBEDDINGS_PATH = "./src/embeddings"
METADATA_PATH = "./database.json"

def crear_estructura_directorios():
    """Crea la estructura de directorios necesaria."""
    os.makedirs(BASE_EMBEDDINGS_PATH, exist_ok=True)
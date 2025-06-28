import os
import json
from config import METADATA_PATH

def cargar_database():
    """Carga la base de datos existente o crea una nueva."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "personas": {},
            "estadisticas": {
                "total_personas": 0,
                "total_embeddings": 0
            }
        }

def guardar_database(database):
    """Guarda la base de datos en formato JSON."""
    with open(METADATA_PATH, "w", encoding='utf-8') as f:
        json.dump(database, f, indent=4, ensure_ascii=False)

def obtener_estadisticas():
    """Obtiene estadísticas detalladas de la base de datos."""
    database = cargar_database()
    
    estadisticas = {
        "resumen": database["estadisticas"],
        "por_persona": {}
    }
    
    for nombre, info in database["personas"].items():
        estadisticas["por_persona"][nombre] = {
            "embeddings": info["total_embeddings"],
            "ruta": info["ruta_embeddings"]
        }
    
    return estadisticas
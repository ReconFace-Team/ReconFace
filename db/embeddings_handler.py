# embeddings_handler.py
import os
import numpy as np
import glob
from config import BASE_EMBEDDINGS_PATH
from database_manager import cargar_database

def obtener_embeddings_persona(nombre_persona):
    """Obtiene todos los embeddings de una persona específica."""
    persona_path = os.path.join(BASE_EMBEDDINGS_PATH, nombre_persona)
    
    if not os.path.exists(persona_path):
        return []
    
    # Buscar todos los archivos .npy en la carpeta de la persona
    npy_files = glob.glob(os.path.join(persona_path, "*.npy"))
    embeddings = []
    
    for file_path in npy_files:
        try:
            embedding = np.load(file_path)
            embeddings.append({
                "archivo": os.path.basename(file_path),
                "embedding": embedding.tolist(),
                "dimensiones": embedding.shape
            })
        except Exception as e:
            print(f"⚠ Error cargando {file_path}: {e}")
    
    return embeddings

def obtener_todos_los_embeddings():
    """
    Recolecta TODOS los embeddings de todas las personas para reconocimiento facial.
    Retorna: diccionario con arrays numpy y metadata asociada
    """
    database = cargar_database()
    todos_embeddings = {
        'embeddings': [],
        'labels': [],
        'personas': [],
        'archivos': []
    }
    
    print("🔄 Cargando embeddings para reconocimiento facial...")
    
    for nombre, info in database["personas"].items():
        persona_path = info["ruta_embeddings"]
        
        if not os.path.exists(persona_path):
            print(f"⚠ Ruta no encontrada: {persona_path}")
            continue
            
        # Cargar todos los archivos .npy de esta persona
        npy_files = glob.glob(os.path.join(persona_path, "*.npy"))
        
        for file_path in npy_files:
            try:
                embedding = np.load(file_path)
                
                # Validar dimensiones del embedding
                if len(embedding.shape) == 1 and embedding.shape[0] == 128:
                    todos_embeddings['embeddings'].append(embedding)
                    todos_embeddings['labels'].append(nombre)
                    todos_embeddings['personas'].append(info)
                    todos_embeddings['archivos'].append(os.path.basename(file_path))
                else:
                    print(f"⚠ Embedding con dimensiones incorrectas en {file_path}: {embedding.shape}")
                    
            except Exception as e:
                print(f"❌ Error cargando {file_path}: {e}")
    
    # Convertir lista de embeddings a array numpy para eficiencia
    if todos_embeddings['embeddings']:
        todos_embeddings['embeddings'] = np.vstack(todos_embeddings['embeddings'])
        
    print(f"✔ Cargados {len(todos_embeddings['labels'])} embeddings de {len(database['personas'])} personas")
    
    return todos_embeddings

def obtener_embeddings_persona_para_reconocimiento(nombre_persona):
    """
    Obtiene los embeddings de una persona específica en formato optimizado para reconocimiento.
    """
    database = cargar_database()
    
    if nombre_persona not in database["personas"]:
        print(f"❌ Persona '{nombre_persona}' no encontrada")
        return None
    
    info = database["personas"][nombre_persona]
    persona_path = info["ruta_embeddings"]
    
    if not os.path.exists(persona_path):
        print(f"❌ Ruta no encontrada: {persona_path}")
        return None
    
    embeddings = []
    archivos = []
    
    npy_files = glob.glob(os.path.join(persona_path, "*.npy"))
    
    for file_path in npy_files:
        try:
            embedding = np.load(file_path)
            if len(embedding.shape) == 1 and embedding.shape[0] == 128:
                embeddings.append(embedding)
                archivos.append(os.path.basename(file_path))
        except Exception as e:
            print(f"❌ Error cargando {file_path}: {e}")
    
    if embeddings:
        embeddings = np.vstack(embeddings)
        
    return {
        'persona': nombre_persona,
        'embeddings': embeddings,
        'archivos': archivos,
        'total': len(archivos)
    }
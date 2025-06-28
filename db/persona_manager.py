# persona_manager.py
import os
import uuid
import numpy as np
import glob
import shutil
from config import BASE_EMBEDDINGS_PATH, crear_estructura_directorios
from database_manager import cargar_database, guardar_database
from embeddings_handler import obtener_embeddings_persona

def almacenar_persona(nombre, ocupacion, embeddings_array=None):
    """
    Registra una nueva persona en la base de datos.
    
    Args:
        nombre (str): Nombre de la persona
        ocupacion (str): Ocupación de la persona
        embeddings_array (list, optional): Lista de arrays numpy con embeddings
    """
    crear_estructura_directorios()
    
    # Cargar base de datos existente
    database = cargar_database()
    
    # Validar que la persona no exista ya
    if nombre in database["personas"]:
        print(f"⚠ La persona '{nombre}' ya existe en la base de datos")
        return False
    
    # Crear directorio para la persona
    persona_path = os.path.join(BASE_EMBEDDINGS_PATH, nombre)
    os.makedirs(persona_path, exist_ok=True)
    
    # Generar ID único
    persona_id = str(uuid.uuid4())
    
    # Si se proporcionan embeddings, los guardamos
    if embeddings_array:
        print('Hay array.')
        for i, embedding in enumerate(embeddings_array):
            if isinstance(embedding, np.ndarray):
                archivo_path = os.path.join(persona_path, f"embedding_{i:04d}.npy")
                np.save(archivo_path, embedding)

    # Contamos todos los embeddings existentes en la carpeta
    embeddings_existentes = len(glob.glob(os.path.join(persona_path, "*.npy")))

    # Si no hay ninguno, abortamos
    if embeddings_existentes == 0:
        print(f"⚠ No se encontraron embeddings para '{nombre}' en {persona_path}")
        return False
    
    # Registrar persona en la base de datos
    database["personas"][nombre] = {
        "id": persona_id,
        "nombre": nombre,
        "ruta_embeddings": persona_path,
        "total_embeddings": embeddings_existentes,
        "fecha_registro": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
    }
    
    # Actualizar estadísticas
    database["estadisticas"]["total_personas"] += 1
    database["estadisticas"]["total_embeddings"] += embeddings_existentes
    
    # Guardar base de datos
    guardar_database(database)
    
    print(f"✔ Registro exitoso: {nombre} ({ocupacion})")
    print(f"  - ID: {persona_id}")
    print(f"  - Embeddings: {embeddings_existentes}")
    print(f"  - Ruta: {persona_path}")
    
    return True

def buscar_persona(nombre):
    """Busca una persona en la base de datos y retorna su información completa."""
    database = cargar_database()
    
    if nombre not in database["personas"]:
        print(f"❌ Persona '{nombre}' no encontrada")
        return None
    
    persona_info = database["personas"][nombre].copy()
    
    # Cargar embeddings
    embeddings = obtener_embeddings_persona(nombre)
    persona_info["embeddings"] = embeddings
    
    return persona_info

def listar_personas():
    """Lista todas las personas registradas."""
    database = cargar_database()
    
    print("\n=== BASE DE DATOS DE PERSONAS ===")
    print(f"Total personas: {database['estadisticas']['total_personas']}")
    print(f"Total embeddings: {database['estadisticas']['total_embeddings']}")
    print("\nPersonas registradas:")
    
    for nombre, info in database["personas"].items():
        print(f"  • {nombre} - {info['total_embeddings']} embeddings")
    
    return database["personas"]

def actualizar_conteo_embeddings():
    """Actualiza el conteo de embeddings para todas las personas."""
    database = cargar_database()
    total_embeddings = 0
    
    for nombre, info in database["personas"].items():
        persona_path = info["ruta_embeddings"]
        if os.path.exists(persona_path):
            count = len(glob.glob(os.path.join(persona_path, "*.npy")))
            database["personas"][nombre]["total_embeddings"] = count
            total_embeddings += count
        else:
            print(f"⚠ Ruta no encontrada para {nombre}: {persona_path}")
    
    database["estadisticas"]["total_embeddings"] = total_embeddings
    guardar_database(database)
    
    print(f"✔ Conteo actualizado: {total_embeddings} embeddings totales")

def eliminar_persona(nombre):
    """Elimina una persona y todos sus embeddings."""
    database = cargar_database()
    
    if nombre not in database["personas"]:
        print(f"❌ Persona '{nombre}' no encontrada")
        return False
    
    # Eliminar directorio y archivos
    persona_path = database["personas"][nombre]["ruta_embeddings"]
    if os.path.exists(persona_path):
        shutil.rmtree(persona_path)
        print(f"✔ Eliminados archivos de {persona_path}")
    
    # Eliminar de la base de datos
    del database["personas"][nombre]
    
    # Actualizar estadísticas
    database["estadisticas"]["total_personas"] -= 1
    
    # Recalcular total de embeddings
    total_embeddings = sum(info["total_embeddings"] for info in database["personas"].values())
    database["estadisticas"]["total_embeddings"] = total_embeddings
    
    guardar_database(database)
    print(f"✔ Persona '{nombre}' eliminada completamente")
    
    return True
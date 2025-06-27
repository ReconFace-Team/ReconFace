import os
import uuid
import json
import numpy as np

# Archivos de almacenamiento
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "metadata.json"

# Cargar embeddings existentes o crear uno vacío
if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    embeddings = np.empty((0, 128), dtype='float32')

# Cargar metadata existente o crear una lista vacía
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
else:
    metadata = []

def almacenar_persona(nombre, ocupacion, embedding_vector):
    """Guarda un embedding facial junto con nombre y ocupación."""
    if len(embedding_vector) != 128:
        raise ValueError("El embedding debe tener exactamente 128 dimensiones")

    # Asignar ID único
    persona_id = str(uuid.uuid4())

    # Agregar nuevo embedding
    global embeddings, metadata
    embeddings = np.vstack([embeddings, embedding_vector])

    # Agregar metadata asociada
    metadata.append({
        "id": persona_id,
        "nombre": nombre,
        "ocupacion": ocupacion
    })

    # Guardar archivos
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✔ Registro exitoso: {nombre} ({ocupacion}) — ID: {persona_id}")

# recognition_engine.py
import numpy as np
from embeddings_handler import obtener_todos_los_embeddings, obtener_embeddings_persona_para_reconocimiento
from database_manager import cargar_database

def crear_dataset_reconocimiento(personas_incluir=None, limite_por_persona=None):
    """
    Crea un dataset optimizado para entrenar/usar modelos de reconocimiento facial.
    
    Args:
        personas_incluir (list): Lista de nombres de personas a incluir. Si es None, incluye todas.
        limite_por_persona (int): Límite máximo de embeddings por persona. Si es None, incluye todos.
    
    Returns:
        dict: Dataset con embeddings, labels y metadata
    """
    database = cargar_database()
    
    dataset = {
        'X': [],  # Array de embeddings
        'y': [],  # Labels numéricas
        'personas': [],  # Nombres de personas
        'label_map': {},  # Mapeo label_numérica -> nombre_persona
        'persona_map': {}  # Mapeo nombre_persona -> label_numérica
    }
    
    # Determinar qué personas incluir
    if personas_incluir is None:
        personas_incluir = list(database["personas"].keys())
    
    print(f"🔄 Creando dataset con {len(personas_incluir)} personas...")
    
    label_counter = 0
    
    for nombre in personas_incluir:
        if nombre not in database["personas"]:
            print(f"⚠ Persona '{nombre}' no encontrada, omitiendo...")
            continue
            
        # Obtener embeddings de esta persona
        persona_data = obtener_embeddings_persona_para_reconocimiento(nombre)
        
        if persona_data is None or len(persona_data['embeddings']) == 0:
            print(f"⚠ No hay embeddings válidos para '{nombre}', omitiendo...")
            continue
        
        # Aplicar límite si se especifica
        embeddings = persona_data['embeddings']
        if limite_por_persona and len(embeddings) > limite_por_persona:
            # Seleccionar aleatoriamente
            indices = np.random.choice(len(embeddings), limite_por_persona, replace=False)
            embeddings = embeddings[indices]
        
        # Agregar al dataset
        for embedding in embeddings:
            dataset['X'].append(embedding)
            dataset['y'].append(label_counter)
            dataset['personas'].append(nombre)
        
        # Mapeos
        dataset['label_map'][label_counter] = nombre
        dataset['persona_map'][nombre] = label_counter
        
        print(f"  ✔ {nombre}: {len(embeddings)} embeddings agregados (label: {label_counter})")
        label_counter += 1
    
    # Convertir a arrays numpy
    if dataset['X']:
        dataset['X'] = np.vstack(dataset['X'])
        dataset['y'] = np.array(dataset['y'])
    
    print(f"✔ Dataset creado: {len(dataset['X'])} embeddings de {len(dataset['label_map'])} personas")
    
    return dataset

def buscar_persona_por_embedding(embedding_query, umbral_similitud=0.7):
    """
    Busca la persona más similar a un embedding dado usando similitud coseno.
    
    Args:
        embedding_query (np.array): Embedding de consulta (128 dimensiones)
        umbral_similitud (float): Umbral mínimo de similitud para considerar una coincidencia
    
    Returns:
        dict: Información de la persona más similar o None si no hay coincidencias
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Cargar todos los embeddings
    todos_embeddings = obtener_todos_los_embeddings()
    
    if len(todos_embeddings['embeddings']) == 0:
        print("❌ No hay embeddings en la base de datos")
        return None
    
    # Calcular similitudes
    embedding_query = embedding_query.reshape(1, -1)
    similitudes = cosine_similarity(embedding_query, todos_embeddings['embeddings'])[0]
    
    # Encontrar la mejor coincidencia
    mejor_indice = np.argmax(similitudes)
    mejor_similitud = similitudes[mejor_indice]
    
    if mejor_similitud < umbral_similitud:
        print(f"❌ No se encontró coincidencia (similitud máxima: {mejor_similitud:.3f})")
        return None
    
    persona_nombre = todos_embeddings['labels'][mejor_indice]
    persona_info = todos_embeddings['personas'][mejor_indice]
    
    resultado = {
        'persona': persona_nombre,
        'similitud': mejor_similitud,
        'archivo_origen': todos_embeddings['archivos'][mejor_indice],
        'confianza': 'Alta' if mejor_similitud > 0.9 else 'Media' if mejor_similitud > 0.8 else 'Baja'
    }
    
    print(f"✔ Persona identificada: {persona_nombre} (similitud: {mejor_similitud:.3f})")
    
    return resultado
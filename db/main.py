
# main.py
from persona_manager import almacenar_persona
from embeddings_handler import obtener_embeddings_persona

# Ejemplo de uso para reconocimiento facial
if __name__ == "__main__":
    nombres = ['jhoan aedo', 'ariel del rio', 'benjamin valdivia']

    for name in nombres:
        embedding_user = obtener_embeddings_persona(name)

        almacenar_persona(name, embedding_user)
import numpy as np

# Ajusta este import si tu ruta es distinta
from main.recognition.face_recognizer import OptimizedFaceRecognizer
from main.recognition.face_processor import FaceProcessor


def test_recognition_pipeline_initialization():
    """
    Verifica que el reconocedor y el procesador de caras se inicializan
    correctamente y que se cargan embeddings.
    """
    recognizer = OptimizedFaceRecognizer()
    assert len(recognizer.face_embeddings) > 0, "No se cargaron embeddings"

    processor = FaceProcessor(recognizer)
    assert processor is not None, "No se pudo inicializar FaceProcessor"


def test_recognition_process_frame_with_blank_image():
    """
    Caja blanca suave: se prueba que el pipeline de reconocimiento
    soporta una imagen vacía (sin caras) sin lanzar errores.
    """
    recognizer = OptimizedFaceRecognizer()
    processor = FaceProcessor(recognizer)

    # Imagen vacía (negra) 480x640
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = processor.process_frame(dummy_frame)

    # No esperamos caras, pero sí que devuelva una lista y que no falle
    assert isinstance(results, list), "El resultado debe ser una lista"
    # No es obligatorio que la lista esté vacía, pero en la práctica debería:
    # assert len(results) == 0

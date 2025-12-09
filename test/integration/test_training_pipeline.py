import os
from pathlib import Path

# Ajusta estos imports según tu estructura real
from main.training.face_processor import FaceProcessor
from main.training.config import INPUT_DIR, OUTPUT_DIR, MAX_WIDTH, MAX_HEIGHT


def _get_any_image_path() -> str:
    """
    Devuelve la ruta de una imagen cualquiera dentro de INPUT_DIR
    para usarla en la prueba de integración.
    """
    root = Path(INPUT_DIR)
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                return str(Path(dirpath) / fname)
    return ""


def test_training_single_image_processing():
    """
    Prueba de integración: verifica que el pipeline de entrenamiento
    es capaz de procesar al menos una imagen sin errores y generar
    al menos un embedding en OUTPUT_DIR.
    """
    img_path = _get_any_image_path()
    assert img_path != "", (
        "No se encontraron imágenes en INPUT_DIR. "
        "Coloca al menos una imagen en src/images/<persona>/ para esta prueba."
    )

    filename = os.path.basename(img_path)

    processor = FaceProcessor()

    # Ejecutar el pipeline de una sola imagen
    success = processor.process_single_image(
        img_path=img_path,
        filename=filename,
        output_dir=OUTPUT_DIR,
        max_width=MAX_WIDTH,
        max_height=MAX_HEIGHT,
    )

    assert success, "El procesamiento de la imagen de entrenamiento falló"

    # Verificar que en OUTPUT_DIR exista una carpeta para la persona
    # y al menos un archivo .npy (embedding)
    person_name = Path(os.path.dirname(img_path)).relative_to(INPUT_DIR).parts[0]
    person_dir = Path(OUTPUT_DIR) / person_name
    assert person_dir.exists(), f"No se creó el directorio de embeddings para {person_name}"

    npy_files = [f for f in os.listdir(person_dir) if f.endswith(".npy")]
    assert len(npy_files) > 0, "No se generaron embeddings (.npy) en la carpeta de la persona"

import os
import sys
from pathlib import Path

# -------------------------------------------------------------------
# Asegurar que se puedan importar m贸dulos del MISMO directorio
# cuando se ejecuta como script: python main/training/main.py
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent          # .../main/training
PROJECT_ROOT = CURRENT_DIR.parents[1]                  # .../ReconFace-openVINO

# A帽adir carpeta de entrenamiento al sys.path
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# A帽adir ra铆z del proyecto al sys.path (para importar main.recognition)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Imports de config (training/config.py) - robusto
# -------------------------------------------------------------------
try:
    # Cuando se ejecuta como script (python main/training/main.py)
    from config import (
        INPUT_DIR,
        OUTPUT_DIR,
        MAX_WIDTH,
        MAX_HEIGHT,
        MIN_DET_SCORE,
        CHECK_EXISTING_EMBEDDINGS,
    )
except ImportError:
    # Cuando se usa como paquete (python -m main.training.main)
    from .config import (
        INPUT_DIR,
        OUTPUT_DIR,
        MAX_WIDTH,
        MAX_HEIGHT,
        MIN_DET_SCORE,
        CHECK_EXISTING_EMBEDDINGS,
    )

# -------------------------------------------------------------------
# Imports locales (face_processor, utils) - robustos
# -------------------------------------------------------------------
try:
    from face_processor import FaceProcessor
except ImportError:
    from .face_processor import FaceProcessor

try:
    from utils import (
        is_image_file,
        print_final_tips,
        count_existing_embeddings,
    )
except ImportError:
    from .utils import (
        is_image_file,
        print_final_tips,
        count_existing_embeddings,
    )

# -------------------------------------------------------------------
# PerformanceMonitor para m茅tricas de ENTRENAMIENTO
# (usa main/recognition/perf_monitor.py)
# -------------------------------------------------------------------
from main.recognition.perf_monitor import PerformanceMonitor


def print_existing_embeddings_summary():
    """Print summary of existing embeddings"""
    if not os.path.exists(OUTPUT_DIR):
        print("馃搧 No existing embeddings directory found")
        return

    print("馃搳 Existing Embeddings Summary:")
    total_embeddings = 0

    for person_folder in os.listdir(OUTPUT_DIR):
        person_path = os.path.join(OUTPUT_DIR, person_folder)
        if os.path.isdir(person_path):
            count = count_existing_embeddings(person_path)
            if count > 0:
                print(f"   {person_folder}: {count} embeddings")
                total_embeddings += count

    print(f"   Total existing embeddings: {total_embeddings}\n")


def main():
    """Main function to process all images in subdirectories"""

    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Monitor de rendimiento en fase 'training'
    # (usa src/metrics/training/perf_YYYYMMDD.jsonl)
    try:
        from main.recognition import config as rec_cfg
        interval = int(getattr(rec_cfg, "PERF_MONITOR_INTERVAL_SEC", 5))
    except Exception:
        interval = 5

    perf_monitor = PerformanceMonitor(interval_sec=interval, phase="training")

    # Mostrar resumen de embeddings existentes (si est谩 habilitado)
    if CHECK_EXISTING_EMBEDDINGS:
        print_existing_embeddings_summary()

    # Inicializar FaceProcessor
    processor = FaceProcessor()

    # Contadores
    processed_count = 0
    skipped_count = 0
    total_count = 0

    # Procesar todas las im谩genes en subdirectorios de INPUT_DIR
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if not is_image_file(filename):
                continue

            total_count += 1
            img_path = os.path.join(root, filename)

            success = processor.process_single_image(
                img_path, filename, OUTPUT_DIR, MAX_WIDTH, MAX_HEIGHT
            )

            # Determinar si la imagen se proces贸 o se salt贸
            is_skipped = False
            if success:
                # Hay implementaciones donde 'success' puede ser un mensaje
                # que incluye "skipping" si fue omitida.
                if "skipping" in str(success).lower():
                    skipped_count += 1
                    is_skipped = True
                else:
                    processed_count += 1
            else:
                skipped_count += 1
                is_skipped = True

            # --- Registrar m茅trica en el monitor de entrenamiento -------
            # known_faces   -> 1 si se proces贸 realmente, 0 si se salt贸
            # unknown_faces -> 0 (no aplica aqu铆, pero lo dejamos por consistencia)
            # frames        -> 1 unidad de trabajo (1 imagen evaluada)
            try:
                if is_skipped:
                    perf_monitor.tick(num_known=0, num_unknown=0, frames=1)
                else:
                    perf_monitor.tick(num_known=1, num_unknown=0, frames=1)
            except Exception as e:
                # No queremos que falle el entrenamiento por un problema de m茅tricas
                print(f"[TRAIN] Warning in perf_monitor.tick(): {e}")
            # ------------------------------------------------------------

    # Estad铆sticas finales
    print(f"\n馃搳 Final Statistics:")
    print(f"   Total images found: {total_count}")
    print(f"   Successfully processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")

    # Resumen final de embeddings
    if CHECK_EXISTING_EMBEDDINGS:
        print("\n" + "=" * 50)
        print_existing_embeddings_summary()

    print_final_tips(MIN_DET_SCORE)


if __name__ == "__main__":
    main()
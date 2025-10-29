# === CONFIGURATION SETTINGS ===
from pathlib import Path

# --- Rutas base absolutas ---
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]     # ...\ReconFace-main\ReconFace-main
MODELS_DIR   = PROJECT_ROOT / "models"

INPUT_DIR  = str(PROJECT_ROOT / "src" / "images")
OUTPUT_DIR = str(PROJECT_ROOT / "src" / "embeddings")

# --- Modelos OpenVINO (IR/OMZ) ---
OV_DET_MODEL = str(MODELS_DIR / "intel"  / "face-detection-0200"                     / "FP32" / "face-detection-0200.xml")
OV_LMK_MODEL = str(MODELS_DIR / "intel"  / "landmarks-regression-retail-0009"        / "FP32" / "landmarks-regression-retail-0009.xml")
OV_REC_MODEL = str(MODELS_DIR / "public" / "face-recognition-resnet100-arcface-onnx" / "FP32" / "face-recognition-resnet100-arcface-onnx.xml")

# === OBJETIVO POR PERSONA ===
TARGET_EMB_PER_PERSON = 1000   # total deseado por persona (original + augmentations)

# Límites de augs por imagen (para repartir el faltante)
AUGS_PER_IMAGE_MIN = 20
AUGS_PER_IMAGE_MAX = 200

# Augmentation settings (fallback si no se usa el dinámico)
N_AUGMENTATIONS = 100

# === MULTIESCALA (simular distancia) ===
# Proporción de augmentations por bucket (suma ≈ 1.0)
SCALE_BUCKETS = {
    "small":  0.34,   # cara más pequeña (simula lejos)
    "medium": 0.33,   # cara normal
    "large":  0.33    # cara más grande (simula cerca)
}
# Parámetros de render por bucket:
SMALL_CANVAS_MULT = 1.7   # 1.5–2.0: sube para que la cara quede aún más chica (más “lejos”)
LARGE_ZOOM        = 1.6   # 1.3–2.0: sube para que la cara quede aún más grande (más “cerca”)

# === POST-PROCESS: ELIMINAR IMÁGENES FUENTE ===
DELETE_SOURCE_IMAGES     = True   # borrar la imagen fuente cuando se terminen los embeddings de esa imagen
DELETE_REQUIRE_FULL_AUGS = True   # borrar solo si se alcanzó el objetivo de augs (dinámico o N_AUGMENTATIONS)
MOVE_TO_TRASH            = False  # PERMANENTE (no papelera)

# === Image processing settings ===
MAX_WIDTH  = 1000
MAX_HEIGHT = 1000
MIN_DET_SCORE = 0.70

# === Preview settings ===
SHOW_PREVIEW      = True
PREVIEW_DURATION  = 2000  # ms
PREVIEW_INTERVAL  = 10    # mostrar preview cada N embeddings exitosos

# === Processing settings ===
MAX_ATTEMPTS                 = 50   # intentos máx por augmentation
REGENERATE_TRANSFORM_INTERVAL = 25  # (si usas transform persistente)
LOG_INTERVAL                 = 10   # logs cada N intentos

# === Embedding checking settings ===
CHECK_EXISTING_EMBEDDINGS = True  # salta imágenes/personas si ya existen embeddings suficientes
MIN_EMBEDDINGS_THRESHOLD  = 50    # (no se usa en la versión dinámica, lo mantenemos por compatibilidad)
SKIP_COMPLETED_IMAGES     = True  # salta si ya están todos los augs de esa imagen

# === OPENVINO SETTINGS (TRAINING) ===
OV_DEVICE = "CPU"                  # "CPU", "GPU" o "AUTO"
OV_PERFORMANCE_HINT = "LATENCY"    # "THROUGHPUT" o "LATENCY"

# Entradas de modelos
LMK_INPUT_SIZE = (48, 48)          # h, w
REC_INPUT_SIZE = (112, 112)        # h, w
EXPECTED_COLOR_ORDER = "BGR"       # OMZ/IR usa BGR

# === ORIGINAL DETECTION RETRIES (para la imagen original) ===
ORIG_DET_RETRIES       = 10
ORIG_SCALES            = [1.0, 1.2, 0.8, 1.4, 0.6, 1.6, 0.7]
ORIG_USE_ENHANCERS     = True      # aplicar CLAHE, sharpen y gamma en los reintentos
ORIG_PREVIEW_EACH_HIT  = True      # mostrar preview cada vez que encuentre caras válidas

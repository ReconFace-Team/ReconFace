"""
Configuration settings for the face recognition system
"""

from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# === EMBEDDING CONFIGURATION ===
EMBEDDING_DIR = str(PROJECT_ROOT / "src" / "embeddings")
THRESHOLD = 0.50           # Más estricto para mayor precisión con 1000 embeddings
MIN_CONFIDENCE = 0.85      # Aumentado para mejor calidad
TEMPORAL_WINDOW = 7        # Mayor ventana temporal para más estabilidad
MIN_FACE_SIZE = 30         # Reducido para larga distancia
QUALITY_THRESHOLD = 0.7    # Calidad mínima de embedding

# === PERFORMANCE OPTIMIZATION ===
PROCESS_EVERY_N_FRAMES = 1  # Procesar cada frame
FRAME_RESIZE_FACTOR = 1.0   # Sin redimensionar para preservar resolución
BATCH_SIZE = 4              # Procesar múltiples caras en lote

# === LONG DISTANCE CONFIGURATION ===
ENABLE_SUPER_RESOLUTION = True
DISTANCE_ADAPTIVE_THRESHOLD = True
ENHANCED_PREPROCESSING = True

# === CAMERA CONFIGURATION ===
USE_RTSP = 0  # 0 = cámara local, 1 = RTSP remoto
RTSP_URL = "rtsp://100.91.199.92:8554/webcam"
CAMERA_INDEX = 0  # Índice de cámara local

# === CAMERA SETTINGS ===
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 120
CAMERA_BUFFER_SIZE = 1
CAMERA_BACKEND = "MSMF"
PREFER_MJPEG = True

# === FACE ANALYSIS SETTINGS ===
DETECTION_SIZE = (640, 640)
CTX_ID = 0  # -1 CPU, 0+ GPU

# === TEMPORAL SMOOTHING ===
MIN_TEMPORAL_CONSISTENCY = 3
MIN_TEMPORAL_CONSISTENCY_UNKNOWN = 2

# === ADAPTIVE THRESHOLD SETTINGS ===
BASE_FACE_SIZE = 80
MEDIUM_FACE_SIZE = 50
SMALL_FACE_SIZE = 30
THRESHOLD_ADJUSTMENT_MEDIUM = 0.05
THRESHOLD_ADJUSTMENT_SMALL = 0.10
THRESHOLD_ADJUSTMENT_VERY_SMALL = 0.15

# === IMAGE ENHANCEMENT ===
CONTRAST_ALPHA = 1.2
BRIGHTNESS_BETA = 15
BILATERAL_FILTER_D = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75
SUPER_RESOLUTION_SCALE = 2.0
SUPER_RESOLUTION_THRESHOLD = 100

# === DISPLAY SETTINGS ===
HIGH_CONFIDENCE_THRESHOLD = 85
MEDIUM_CONFIDENCE_THRESHOLD = 70
LOW_CONFIDENCE_THRESHOLD = 50

# === COLORS (BGR) ===
COLOR_HIGH_CONFIDENCE = (0, 255, 0)
COLOR_MEDIUM_CONFIDENCE = (0, 255, 255)
COLOR_LOW_CONFIDENCE = (0, 165, 255)
COLOR_VERY_LOW_CONFIDENCE = (0, 100, 255)
COLOR_UNKNOWN = (0, 0, 255)

# === TEXT SETTINGS ===
FONT_SCALE = 0.5
FONT_THICKNESS = 2
FONT_TYPE = 1  # cv2.FONT_HERSHEY_SIMPLEX

# === FAISS INDEX SETTINGS ===
TOP_K_CANDIDATES = 20
CONSISTENCY_THRESHOLD_NORMAL = 15
CONSISTENCY_THRESHOLD_PERMISSIVE = 20
DYNAMIC_THRESHOLD_REDUCTION = 2

# === OPENVINO SETTINGS ===
OV_DEVICE = "CPU"  # "CPU", "GPU", "AUTO"
OV_PERFORMANCE_HINT = "LATENCY"

OV_DET_MODEL = str(MODELS_DIR / "intel"  / "face-detection-0200"                     / "FP32" / "face-detection-0200.xml")
OV_LMK_MODEL = str(MODELS_DIR / "intel"  / "landmarks-regression-retail-0009"        / "FP32" / "landmarks-regression-retail-0009.xml")
OV_REC_MODEL = str(MODELS_DIR / "public" / "face-recognition-resnet100-arcface-onnx" / "FP32" / "face-recognition-resnet100-arcface-onnx.xml")

DET_MIN_CONFIDENCE = 0.60
LMK_INPUT_SIZE = (48, 48)
REC_INPUT_SIZE = (112, 112)
EXPECTED_COLOR_ORDER = "BGR"

# === AUTOLEARN (reconocimiento) ===
AUTOLEARN_ENABLED = True

# Tiempo mínimo desde el enrolamiento (en días)
AUTOLEARN_MIN_DAYS_SINCE_ENROLL = 14

# Criterios de confianza (porcentaje 0..100)
AUTOLEARN_QUARANTINE_MIN_CONF_PCT = 95.0   # encola frames si >= 95
AUTOLEARN_PROMOTE_MIN_CONF_PCT    = 97.5   # promueve si >= 97.5

# Embeddings a generar al promover un frame
AUTOLEARN_BATCH_EMBS = 100

# Cada cuántos frames intentar procesar la cuarentena
AUTOLEARN_PROCESS_EVERY_N_FRAMES = 30

# === RUTAS DE AUTOLEARN ===
AUTOLEARN_DIR = Path(EMBEDDING_DIR).parent / "autolearn"
AUTOLEARN_QUARANTINE_DIR       = str(AUTOLEARN_DIR / "quarantine")
AUTOLEARN_ALT_EMBEDDINGS_DIR   = str(AUTOLEARN_DIR / "embeddings_alt")
AUTOLEARN_DB_PATH              = str(AUTOLEARN_DIR / "autolearn_meta.jsonl")

# Crear directorios si no existen (por seguridad)
for _path in [AUTOLEARN_DIR, Path(AUTOLEARN_QUARANTINE_DIR), Path(AUTOLEARN_ALT_EMBEDDINGS_DIR)]:
    _path.mkdir(parents=True, exist_ok=True)

"""
Configuration settings for the face recognition system (reconocimiento + GUI)
"""

from pathlib import Path

# =====================================================================================
#                                       PATHS
# =====================================================================================

# Este archivo está en: PROJECT_ROOT/main/recognition/config.py
# parents[0] = .../main/recognition
# parents[1] = .../main
# parents[2] = .../ReconFace-main   ← lo que queremos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# =====================================================================================
#                           EMBEDDING / RECOGNITION SETTINGS
# =====================================================================================

EMBEDDING_DIR = str(PROJECT_ROOT / "src" / "embeddings")

THRESHOLD = 0.50           # Umbral base para decisión de match (similaridad/distancia)
MIN_CONFIDENCE = 0.80      # Confianza mínima para mostrar como reconocido
TEMPORAL_WINDOW = 7        # Ventana temporal para suavizado (si se usa)
MIN_FACE_SIZE = 30         # Tamaño mínimo de cara (px)
QUALITY_THRESHOLD = 0.7    # Calidad mínima de embedding

# =====================================================================================
#                           PERFORMANCE OPTIMIZATION
# =====================================================================================

PROCESS_EVERY_N_FRAMES = 2  # Procesar cada frame
FRAME_RESIZE_FACTOR = 1   # 1.0 = sin redimensionar
BATCH_SIZE = 4              # Caras por lote

# =====================================================================================
#                           LONG DISTANCE CONFIG
# =====================================================================================

ENABLE_SUPER_RESOLUTION = True
DISTANCE_ADAPTIVE_THRESHOLD = True
ENHANCED_PREPROCESSING = True

# =====================================================================================
#                           CAMERA CONFIGURATION (modo CLI)
# =====================================================================================

# Este bloque se usa en el main de consola, NO en la GUI.
USE_RTSP = False             # False = cámara local, True = RTSP remoto
RTSP_URL = "rtsp://100.91.199.92:8554/webcam"

CAMERA_INDEX = 0            # Índice de cámara local (0,1,2,...)
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 120
CAMERA_BUFFER_SIZE = 1
CAMERA_BACKEND = "ANY"      # "ANY" = probar MSMF → DSHOW → default
PREFER_MJPEG = True

# =====================================================================================
#                           FACE ANALYSIS SETTINGS
# =====================================================================================

DETECTION_SIZE = (640, 640)
CTX_ID = 0  # -1 CPU, 0+ GPU

# Suavizado temporal
MIN_TEMPORAL_CONSISTENCY = 3
MIN_TEMPORAL_CONSISTENCY_UNKNOWN = 2

# Umbral adaptativo según tamaño de cara
BASE_FACE_SIZE = 80
MEDIUM_FACE_SIZE = 50
SMALL_FACE_SIZE = 30
THRESHOLD_ADJUSTMENT_MEDIUM = 0.05
THRESHOLD_ADJUSTMENT_SMALL = 0.10
THRESHOLD_ADJUSTMENT_VERY_SMALL = 0.15

# =====================================================================================
#                           IMAGE ENHANCEMENT
# =====================================================================================

CONTRAST_ALPHA = 1.2
BRIGHTNESS_BETA = 15
BILATERAL_FILTER_D = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75
SUPER_RESOLUTION_SCALE = 2.0
SUPER_RESOLUTION_THRESHOLD = 100

# =====================================================================================
#                           DISPLAY SETTINGS
# =====================================================================================

HIGH_CONFIDENCE_THRESHOLD = 85
MEDIUM_CONFIDENCE_THRESHOLD = 70
LOW_CONFIDENCE_THRESHOLD = 50

COLOR_HIGH_CONFIDENCE = (0, 255, 0)
COLOR_MEDIUM_CONFIDENCE = (0, 255, 255)
COLOR_LOW_CONFIDENCE = (0, 165, 255)
COLOR_VERY_LOW_CONFIDENCE = (0, 100, 255)
COLOR_UNKNOWN = (0, 0, 255)

FONT_SCALE = 0.5
FONT_THICKNESS = 2
FONT_TYPE = 1  # cv2.FONT_HERSHEY_SIMPLEX

# =====================================================================================
#                               FAISS INDEX SETTINGS
# =====================================================================================

TOP_K_CANDIDATES = 20
CONSISTENCY_THRESHOLD_NORMAL = 15
CONSISTENCY_THRESHOLD_PERMISSIVE = 20
DYNAMIC_THRESHOLD_REDUCTION = 2

# =====================================================================================
#                               OPENVINO MODEL PATHS
# =====================================================================================

OV_DEVICE = "CPU"  # "CPU", "GPU", "AUTO"
OV_PERFORMANCE_HINT = "LATENCY"

OV_DET_MODEL = str(
    MODELS_DIR / "intel" / "face-detection-0200" / "FP32" / "face-detection-0200.xml"
)
OV_LMK_MODEL = str(
    MODELS_DIR / "intel" / "landmarks-regression-retail-0009" / "FP32" / "landmarks-regression-retail-0009.xml"
)
OV_REC_MODEL = str(
    MODELS_DIR / "public" / "face-recognition-resnet100-arcface-onnx" / "FP32" / "face-recognition-resnet100-arcface-onnx.xml"
)

DET_MIN_CONFIDENCE = 0.60
LMK_INPUT_SIZE = (48, 48)
REC_INPUT_SIZE = (112, 112)
EXPECTED_COLOR_ORDER = "BGR"

# =====================================================================================
#                                   AUTOLEARN
# =====================================================================================

AUTOLEARN_ENABLED = False

# Tiempo mínimo desde el enrolamiento (en días) para que empiece a auto-aprender
AUTOLEARN_MIN_DAYS_SINCE_ENROLL = 0

# Criterios de confianza (porcentaje 0..100)
AUTOLEARN_QUARANTINE_MIN_CONF_PCT = 95.0   # encola frames si >= 95%
AUTOLEARN_PROMOTE_MIN_CONF_PCT    = 97.5   # promueve si >= 97.5%

# Embeddings a generar al promover un frame
AUTOLEARN_BATCH_EMBS = 100

# Cada cuántos frames intentar procesar la cuarentena
AUTOLEARN_PROCESS_EVERY_N_FRAMES = 30

# Directorios para AutoLearn
AUTOLEARN_DIR = PROJECT_ROOT / "src" / "autolearn"
AUTOLEARN_QUARANTINE_DIR     = str(AUTOLEARN_DIR / "quarantine")
AUTOLEARN_ALT_EMBEDDINGS_DIR = str(AUTOLEARN_DIR / "embeddings_alt")
AUTOLEARN_DB_PATH            = str(AUTOLEARN_DIR / "autolearn_meta.jsonl")

# =====================================================================================
#                                   WHITELIST / BLACKLIST
# =====================================================================================

LISTS_DIR = PROJECT_ROOT / "src" / "lists"
WHITELIST_PATH = str(LISTS_DIR / "whitelist.json")   # personas permitidas
BLACKLIST_PATH = str(LISTS_DIR / "blacklist.json")   # personas bloqueadas

# Estado por defecto al crear una persona nueva desde la GUI
DEFAULT_PERSON_STATUS = "whitelist"

# =====================================================================================
#                                   GUI SETTINGS
# =====================================================================================

GUI_ENABLED = True
GUI_WINDOW_TITLE = "Sistema de Reconocimiento Facial - GUI"
GUI_THEME = "dark"
GUI_MAX_CAMERAS = 4
GUI_REFRESH_INTERVAL_MS = 33  # ~30 FPS

# Fuentes de cámara que la GUI puede usar (estilo sistema de seguridad)
# Ahora configurado para 2 cámaras locales: index=0 (notebook) e index=1 (USB).
GUI_CAMERA_SOURCES = [
    {
        "name": "Notebook (Cam 0)",
        "type": "LOCAL",      # cámara local
        "index": 0,
        "rtsp_url": None,
        "enabled": True,
    },
    {
        "name": "USB (Cam 1)",
        "type": "LOCAL",      # segunda cámara local (USB)
        "index": 1,
        "rtsp_url": None,
        "enabled": True,
    },
    {
        "name": "Cam 3 - RTSP 1",
        "type": "RTSP",
        "index": None,
        "rtsp_url": "rtsp://192.168.0.10:8554/cam2",
        "enabled": False,
    },
    {
        "name": "Cam 4 - RTSP 2",
        "type": "RTSP",
        "index": None,
        "rtsp_url": "rtsp://192.168.0.11:8554/cam3",
        "enabled": False,
    },
]

# =====================================================================================
#                   RUTAS ÚTILES PARA LA VENTANA DE ENTRENAMIENTO (GUI)
# =====================================================================================

# Carpeta donde la GUI dejará las imágenes nuevas para entrenamiento
TRAINING_IMAGES_DIR = str(PROJECT_ROOT / "src" / "images")

# Ruta al script de entrenamiento para invocarlo desde la GUI
TRAINING_SCRIPT_PATH = str(PROJECT_ROOT / "main" / "training" / "main.py")

# =====================================================================================
#                         CREACIÓN AUTOMÁTICA DE DIRECTORIOS
# =====================================================================================

for _path in [
    AUTOLEARN_DIR,
    AUTOLEARN_DIR / "quarantine",
    AUTOLEARN_DIR / "embeddings_alt",
    LISTS_DIR,
]:
    _path.mkdir(parents=True, exist_ok=True)


# =====================================================================================
#                           PERFORMANCE MONITOR (LOG JSON)
# =====================================================================================

PERF_MONITOR_ENABLED = True           # Para activar/desactivar fácil
PERF_MONITOR_INTERVAL_SEC = 5         # Cada cuántos segundos escribe un registro


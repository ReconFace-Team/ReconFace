"""
Configuration settings for the face recognition system
"""

# === EMBEDDING CONFIGURATION ===
EMBEDDING_DIR = "./src/embeddings"
THRESHOLD = 0.50  # Más estricto para mayor precisión con 1000 embeddings
MIN_CONFIDENCE = 0.85  # Aumentado para mejor calidad
TEMPORAL_WINDOW = 7  # Mayor ventana temporal para más estabilidad
MIN_FACE_SIZE = 30  # Reducido para larga distancia
QUALITY_THRESHOLD = 0.7  # Calidad mínima de embedding

# === PERFORMANCE OPTIMIZATION ===
PROCESS_EVERY_N_FRAMES = 1  # Procesar cada frame
FRAME_RESIZE_FACTOR = 1.0  # Sin redimensionar para preservar resolución
BATCH_SIZE = 4  # Procesar múltiples caras en lote

# === LONG DISTANCE CONFIGURATION ===
ENABLE_SUPER_RESOLUTION = True  # Mejorar resolución de caras pequeñas
DISTANCE_ADAPTIVE_THRESHOLD = True  # Threshold dinámico por tamaño
ENHANCED_PREPROCESSING = True  # Preprocesamiento avanzado

# === CAMERA SETTINGS ===
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 120
CAMERA_BUFFER_SIZE = 1

# === FACE ANALYSIS SETTINGS ===
DETECTION_SIZE = (640, 640)  # Tamaño optimizado para detección
CTX_ID = 0  # -1 para CPU, 0+ para GPU

# === TEMPORAL SMOOTHING ===
MIN_TEMPORAL_CONSISTENCY = 3  # Mínimo de detecciones consistentes
MIN_TEMPORAL_CONSISTENCY_UNKNOWN = 2  # Para identidades desconocidas

# === ADAPTIVE THRESHOLD SETTINGS ===
BASE_FACE_SIZE = 80  # Tamaño base para threshold normal
MEDIUM_FACE_SIZE = 50  # Tamaño para caras medianas
SMALL_FACE_SIZE = 30  # Tamaño para caras pequeñas
THRESHOLD_ADJUSTMENT_MEDIUM = 0.05  # Ajuste para caras medianas
THRESHOLD_ADJUSTMENT_SMALL = 0.10  # Ajuste para caras pequeñas
THRESHOLD_ADJUSTMENT_VERY_SMALL = 0.15  # Ajuste para caras muy pequeñas

# === IMAGE ENHANCEMENT ===
CONTRAST_ALPHA = 1.2  # Factor de contraste
BRIGHTNESS_BETA = 15  # Ajuste de brillo
BILATERAL_FILTER_D = 9  # Diámetro del filtro bilateral
BILATERAL_FILTER_SIGMA_COLOR = 75  # Sigma para color
BILATERAL_FILTER_SIGMA_SPACE = 75  # Sigma para espacio
SUPER_RESOLUTION_SCALE = 2.0  # Factor de escala para super-resolución
SUPER_RESOLUTION_THRESHOLD = 100  # Tamaño mínimo para aplicar super-resolución

# === DISPLAY SETTINGS ===
HIGH_CONFIDENCE_THRESHOLD = 85  # Umbral para alta confianza (verde)
MEDIUM_CONFIDENCE_THRESHOLD = 70  # Umbral para confianza media (amarillo)
LOW_CONFIDENCE_THRESHOLD = 50  # Umbral para baja confianza (naranja)

# === COLORS (BGR format for OpenCV) ===
COLOR_HIGH_CONFIDENCE = (0, 255, 0)  # Verde
COLOR_MEDIUM_CONFIDENCE = (0, 255, 255)  # Amarillo
COLOR_LOW_CONFIDENCE = (0, 165, 255)  # Naranja
COLOR_VERY_LOW_CONFIDENCE = (0, 100, 255)  # Naranja oscuro
COLOR_UNKNOWN = (0, 0, 255)  # Rojo

# === TEXT SETTINGS ===
FONT_SCALE = 0.5
FONT_THICKNESS = 2
FONT_TYPE = 1  # cv2.FONT_HERSHEY_SIMPLEX

# === FAISS INDEX SETTINGS ===
TOP_K_CANDIDATES = 20  # Número de candidatos a evaluar
CONSISTENCY_THRESHOLD_NORMAL = 15  # Umbral de consistencia normal
CONSISTENCY_THRESHOLD_PERMISSIVE = 20  # Umbral de consistencia permisivo
DYNAMIC_THRESHOLD_REDUCTION = 2  # Reducción de threshold por embedding
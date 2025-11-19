"""
Camera management for face recognition system
"""

import cv2
import logging
from typing import Optional

from .config import (  # tomamos solo lo que necesitamos
    USE_RTSP,
    RTSP_URL,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_BUFFER_SIZE,
    CAMERA_BACKEND,
)

logger = logging.getLogger(__name__)


def _backend_from_string(name: Optional[str]):
    """
    Convierte el string de CAMERA_BACKEND a la constante de OpenCV.
    Soporta: "MSMF", "DSHOW", "ANY"/None.
    """
    if not name:
        return None
    name = str(name).upper()
    if name == "MSMF":
        return cv2.CAP_MSMF
    if name in ("DSHOW", "DIRECTSHOW"):
        return cv2.CAP_DSHOW
    if name in ("ANY", "DEFAULT"):
        return None
    # fallback: sin backend específico
    return None


class CameraManager:
    """Manages camera initialization and configuration"""

    def __init__(self):
        self.cap = None

    # ---------- helpers internos ----------

    def _try_open_local_with_backend(self, index: int, backend_flag) -> bool:
        """
        Intenta abrir la cámara local con un backend específico
        (o por defecto si backend_flag es None).
        Devuelve True si logra leer un frame válido.
        """
        if backend_flag is None:
            logger.info(f"Intentando abrir cámara local index={index} con backend por defecto")
            cap = cv2.VideoCapture(index)
        else:
            logger.info(f"Intentando abrir cámara local index={index} backend_flag={backend_flag}")
            cap = cv2.VideoCapture(index, backend_flag)

        if not cap.isOpened():
            cap.release()
            return False

        # Configuración básica
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # Buffer pequeño para baja latencia
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        except Exception:
            pass

        # Probar lectura de un frame
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("La cámara se abrió pero no devolvió frame válido.")
            cap.release()
            return False

        # Si llegamos aquí: OK
        self.cap = cap
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Cámara inicializada: {real_w}x{real_h} @ {real_fps:.1f} FPS")
        return True

    def _init_local_camera(self) -> bool:
        """
        Inicializa la cámara local probando:
        1) backend configurado en CAMERA_BACKEND
        2) MSMF
        3) DSHOW
        4) backend por defecto
        """
        index = CAMERA_INDEX
        backend_config = _backend_from_string(CAMERA_BACKEND)

        # Lista de backends a probar en orden (evitando duplicados)
        candidates = [backend_config, cv2.CAP_MSMF, cv2.CAP_DSHOW, None]
        seen = set()
        backends_to_try = []
        for b in candidates:
            key = b if b is not None else "NONE"
            if key not in seen:
                seen.add(key)
                backends_to_try.append(b)

        logger.info(f"Usando cámara local index={index}, backends a probar={backends_to_try}")

        for backend in backends_to_try:
            if self._try_open_local_with_backend(index, backend):
                return True

        logger.error("No se pudo abrir la cámara local con ningún backend.")
        return False

    def _init_rtsp_camera(self) -> bool:
        """Inicializa la cámara RTSP."""
        logger.info("Usando cámara RTSP")
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            logger.error("No se pudo abrir la fuente de video RTSP")
            return False

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        except Exception:
            pass

        self.cap = cap
        logger.info("Cámara RTSP inicializada correctamente")
        return True

    # ---------- API pública usada por el resto del sistema ----------

    def initialize_camera(self):
        """Inicializa la cámara (local o RTSP según config)."""
        if USE_RTSP:
            ok = self._init_rtsp_camera()
        else:
            ok = self._init_local_camera()

        if not ok:
            return False
        return True

    def read_frame(self):
        """Read a frame from the camera"""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Cámara liberada")

    def is_opened(self):
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()

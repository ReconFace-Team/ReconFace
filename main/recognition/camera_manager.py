"""
Camera management for face recognition system
"""

import cv2
import logging

from config import *

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera initialization and configuration"""
    
    def __init__(self):
        self.cap = None
    
    def initialize_camera(self):

        if USE_RTSP:
            logger.info("Usando cámara RTSP")
            self.cap = cv2.VideoCapture(RTSP_URL)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        else:
        # Seleccionar backend en Windows: MSMF (moderno) o DSHOW (legacy)
            backend_flag = 0
            try:
                if CAMERA_BACKEND and CAMERA_BACKEND.upper() == "MSMF":
                    backend_flag = cv2.CAP_MSMF
                elif CAMERA_BACKEND and CAMERA_BACKEND.upper() == "DSHOW":
                    backend_flag = cv2.CAP_DSHOW
            except Exception:
                backend_flag = 0

            logger.info(f"Usando cámara local index={CAMERA_INDEX} backend={CAMERA_BACKEND or 'DEFAULT'}")
            self.cap = cv2.VideoCapture(CAMERA_INDEX, backend_flag) if backend_flag else cv2.VideoCapture(CAMERA_INDEX)

            # Algunas webcams van mejor con MJPG
            try:
                if 'PREFER_MJPEG' in globals() and PREFER_MJPEG:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

            # Propiedades básicas
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

            # Buffer pequeño para baja latencia
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
            except Exception:
                pass

        if not self.cap or not self.cap.isOpened():
            logger.error("No se pudo abrir la fuente de video")
            return False

        # Loguear lo que realmente quedó
        real_w  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Cámara inicializada: {real_w}x{real_h} @ {real_fps:.1f} FPS")
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
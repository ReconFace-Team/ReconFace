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
        """Initialize camera based on configuration"""
        if USE_RTSP:
            logger.info("Usando cámara RTSP")
            self.cap = cv2.VideoCapture(RTSP_URL)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)  # Reducir latencia
        else:
            logger.info("Usando cámara local")
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            logger.error("No se pudo abrir la fuente de video")
            return False
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        logger.info(f"Cámara inicializada: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
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
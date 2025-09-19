"""
Camera management for face recognition system
"""

import cv2
import logging

from config import *

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages multiple camera initialization and configuration"""

    def __init__(self):
        self.cams = []  # List of cv2.VideoCapture objects
        self.indexes = []  # List of detected camera indexes

    def detect_cameras(self, max_tested=10):
        indexes = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                logger.info(f"Camera detected at index {i}")
                indexes.append(i)
            cap.release()
        logger.info(f"Total cameras detected: {len(indexes)}")
        return indexes


    def read_frames(self):
        """
        Read frames from all initialized cameras.
        Returns a list of (success, frame) tuples.
        """
        frames = []
        for cap in self.cams:
            if cap.isOpened():
                ret, frame = cap.read()
                frames.append((ret, frame))
            else:
                frames.append((False, None))
        return frames

    def release_all(self):
        """Release all camera resources"""
        for cap in self.cams:
            if cap is not None:
                cap.release()
        self.cams = []
        self.indexes = []
        logger.info("All cameras released")

    def get_camera_count(self):
        """Return the number of active cameras"""
        return len(self.cams)
    
    def is_opened(self):
        """Check if at least one camera is opened"""
        return any(cap.isOpened() for cap in self.cams)
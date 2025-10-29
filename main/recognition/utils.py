"""
Utility functions for the face recognition system
"""

import logging
import time
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class FPSCounter:
    """FPS calculation utility"""
    
    def __init__(self, window_size=30):
        self.fps_counter = deque(maxlen=window_size)
    
    def update(self, process_time):
        """Update FPS counter with processing time"""
        fps = 1.0 / max(process_time, 0.001)
        self.fps_counter.append(fps)
    
    def get_fps(self):
        """Get current FPS average"""
        if len(self.fps_counter) == 0:
            return 0.0
        return np.mean(self.fps_counter)


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = None
        self.frame_times = []
    
    def start_timing(self):
        """Start timing a frame"""
        self.start_time = time.time()
    
    def end_timing(self):
        """End timing and return duration"""
        if self.start_time is None:
            return 0.0
        
        duration = time.time() - self.start_time
        self.frame_times.append(duration)
        self.start_time = None
        return duration
    
    def get_average_time(self):
        """Get average processing time"""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times)
    
    def reset(self):
        """Reset performance counters"""
        self.frame_times.clear()


def print_statistics(stats, fps_counter):
    """Print detailed statistics"""
    logger.info("=== ESTADÍSTICAS ===")
    logger.info(f"Frames totales: {stats['total_frames']}")
    logger.info(f"Frames procesados: {stats['processed_frames']}")
    logger.info(f"Caras detectadas: {stats['faces_detected']}")
    logger.info(f"Caras reconocidas: {stats['faces_recognized']}")
    
    if stats['faces_detected'] > 0:
        recognition_rate = (stats['faces_recognized'] / stats['faces_detected']) * 100
        logger.info(f"Tasa de reconocimiento: {recognition_rate:.1f}%")
    
    logger.info(f"FPS promedio: {fps_counter.get_fps():.1f}")


def validate_directories():
    """Validate required directories exist"""
    import os
    from config import EMBEDDING_DIR
    
    if not os.path.exists(EMBEDDING_DIR):
        logger.error(f"Directorio {EMBEDDING_DIR} no existe")
        return False
    
    return True
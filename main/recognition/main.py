"""
Threaded multi-camera face recognition system (config-driven)
"""

import cv2
import time
import logging
import numpy as np
import threading

from config import *
from face_recognizer import OptimizedFaceRecognizer
from face_processor import FaceProcessor
from camera_manager import CameraManager
from utils import setup_logging, FPSCounter, PerformanceMonitor, print_statistics, validate_directories

logger = setup_logging()


# ==============================
# Threaded camera capture class
# ==============================
class CameraStream:
    def __init__(self, index, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.running = True
        self.lock = threading.Lock()
        self.fps = 0.0
        self.prev_time = time.time()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                now = time.time()
                dt = now - self.prev_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                self.prev_time = now

    def read(self):
        with self.lock:
            return self.frame.copy(), self.fps

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# ==============================
# Grid creation function
# ==============================
def create_grid(frames, cols, rows, cell_w, cell_h, fps_list):
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(frame, (cell_w, cell_h))
        cv2.putText(resized, f"FPS: {fps_list[idx]:.1f}", (10, 30),
                    FONT_TYPE, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
    return grid


# ==============================
# Main application
# ==============================
def main():
    """Threaded multi-camera face recognition"""
    if not validate_directories():
        return

    logger.info("Inicializando sistema de reconocimiento facial...")

    recognizer = OptimizedFaceRecognizer()
    if len(recognizer.face_embeddings) == 0:
        logger.error("No se encontraron embeddings válidos")
        return

    face_processor = FaceProcessor(recognizer)
    fps_counter = FPSCounter()
    performance_monitor = PerformanceMonitor()

    # Detect connected cameras
    camera_manager = CameraManager()
    cam_indexes = camera_manager.detect_cameras(max_tested=10)
    if len(cam_indexes) == 0:
        logger.error("No se detectaron cámaras")
        return
    logger.info(f"{len(cam_indexes)} cámaras detectadas: {cam_indexes}")

    # Start threaded camera streams
    cams = [CameraStream(idx, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS) for idx in cam_indexes]

    # Compute grid layout
    num_cams = len(cams)
    cols = int(np.ceil(np.sqrt(num_cams)))
    rows = int(np.ceil(num_cams / cols))
    screen_w, screen_h = 1280, 720
    cell_w, cell_h = screen_w // cols, screen_h // rows

    frame_count = 0

    logger.info("Sistema inicializado correctamente")
    logger.info("Controles: 'q' - salir | 's' - estadísticas | 'r' - reset estadísticas")

    try:
        while True:
            frames, fps_list = [], []

            for cam in cams:
                frame, fps = cam.read()
                frames.append(frame)
                fps_list.append(fps)

                # Process every N frames
                frame_count += 5
                # Process frame with processing options
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    recognizer.stats['processed_frames'] += 1
                    results = face_processor.process_frame(frame)  # pass options here if supported
                    frame = face_processor.draw_results(frame, results)  # just draws results


                recognizer.stats['total_frames'] += 1

            # Display grid
            grid = create_grid(frames, cols, rows, cell_w, cell_h, fps_list)
            cv2.imshow("Sistema de Reconocimiento Facial", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Saliendo del sistema...")
                break
            elif key == ord('s') or key == ord('S'):
                print_statistics(recognizer.stats, fps_counter)
            elif key == ord('r') or key == ord('R'):
                logger.info("Reseteando estadísticas...")
                recognizer.stats = {
                    'total_frames': 0,
                    'processed_frames': 0,
                    'faces_detected': 0,
                    'faces_recognized': 0
                }
                performance_monitor.reset()
                frame_count = 0

    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all cameras
        for cam in cams:
            cam.stop()
        cv2.destroyAllWindows()
        logger.info("Sistema cerrado correctamente")


if __name__ == "__main__":
    main()
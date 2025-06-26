"""
Main application for the face recognition system
"""

import cv2
import time
import logging

from config import *
from face_recognizer import OptimizedFaceRecognizer
from face_processor import FaceProcessor
from camera_manager import CameraManager
from utils import setup_logging, FPSCounter, PerformanceMonitor, print_statistics, validate_directories

# Setup logging
logger = setup_logging()


def main():
    """Main application function"""
    # Validate environment
    if not validate_directories():
        return
    
    # Initialize components
    logger.info("Inicializando sistema de reconocimiento facial...")
    
    recognizer = OptimizedFaceRecognizer()
    if len(recognizer.face_embeddings) == 0:
        logger.error("No se encontraron embeddings válidos")
        return
    
    face_processor = FaceProcessor(recognizer)
    camera_manager = CameraManager()
    fps_counter = FPSCounter()
    performance_monitor = PerformanceMonitor()
    
    # Initialize camera
    if not camera_manager.initialize_camera():
        return
    
    logger.info("Sistema inicializado correctamente")
    logger.info("Controles:")
    logger.info("  'q' - Salir")
    logger.info("  's' - Mostrar estadísticas")
    logger.info("  'r' - Reset estadísticas")
    
    frame_count = 0
    
    try:
        while True:
            performance_monitor.start_timing()
            
            ret, frame = camera_manager.read_frame()
            if not ret:
                logger.warning("No se pudo leer el frame")
                break
            
            recognizer.stats['total_frames'] += 1
            frame_count += 1
            
            # Process frame based on configuration
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                recognizer.stats['processed_frames'] += 1
                results = face_processor.process_frame(frame)
                frame = face_processor.draw_results(frame, results)
            
            # Update performance metrics
            process_time = performance_monitor.end_timing()
            fps_counter.update(process_time)
            
            # Display FPS
            current_fps = fps_counter.get_fps()
            cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                       (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Sistema de Reconocimiento Facial", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Saliendo del sistema...")
                break
            elif key == ord('s'):
                print_statistics(recognizer.stats, fps_counter)
            elif key == ord('r'):
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
        # Cleanup
        camera_manager.release()
        cv2.destroyAllWindows()
        logger.info("Sistema cerrado correctamente")


if __name__ == "__main__":
    main()
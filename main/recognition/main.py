"""
Main application for the face recognition system (with AutoLearn)
"""

import cv2
import time
import logging

import config as cfg  # <- usar el módulo de config
from face_recognizer import OptimizedFaceRecognizer
from face_processor import FaceProcessor
from camera_manager import CameraManager
from utils import setup_logging, FPSCounter, PerformanceMonitor, print_statistics, validate_directories

# === AutoLearn (robusto) ===
try:
    import importlib
    m = importlib.import_module("autolearn")
    _AutoLearner = getattr(m, "AutoLearner", None)
    _HAS_AUTOLEARN = callable(_AutoLearner)
    if not _HAS_AUTOLEARN:
        print(f"[AutoLearn] Cargado desde: {getattr(m, '__file__', 'desconocido')}")
        print(f"[AutoLearn] AutoLearner = {type(_AutoLearner)} (esperado: class/callable)")
except Exception:
    _AutoLearner = None
    _HAS_AUTOLEARN = False

# Setup logging
logger = setup_logging()


def draw_autolearn_status(frame, enabled: bool, queue_len: int, x=10, y=55):
    """Pequeño overlay con el estado del AutoLearn."""
    status = "ON" if enabled else "OFF"
    text = f"AutoLearn: {status} | Q:{queue_len}"
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        getattr(cfg, "FONT_SCALE", 0.5),
        (255, 255, 255), 1
    )


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

    # === AutoLearn init ===
    autolearn = None
    auto_enabled = getattr(cfg, "AUTOLEARN_ENABLED", False)
    if _HAS_AUTOLEARN and auto_enabled:
        try:
            autolearn = _AutoLearner()  # usar el símbolo verificado
            autolearn.enabled = True
            logger.info("AutoLearn habilitado")
        except Exception as e:
            logger.error(f"No se pudo inicializar AutoLearn: {e}")
            autolearn = None
    else:
        logger.info("AutoLearn deshabilitado (o no disponible)")

    # Initialize camera
    if not camera_manager.initialize_camera():
        return

    logger.info("Sistema inicializado correctamente")
    logger.info("Controles:")
    logger.info("  'q' - Salir")
    logger.info("  's' - Mostrar estadísticas")
    logger.info("  'r' - Reset estadísticas")
    logger.info("  'a' - Toggle AutoLearn ON/OFF")
    logger.info("  'p' - Forzar procesamiento de cuarentena (AutoLearn)")

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
            if frame_count % getattr(cfg, "PROCESS_EVERY_N_FRAMES", 1) == 0:
                recognizer.stats['processed_frames'] += 1
                results = face_processor.process_frame(frame)
                frame = face_processor.draw_results(frame, results)

                # === AutoLearn: encolar candidatos de alta confianza ===
                if autolearn and autolearn.enabled:
                    min_conf = float(getattr(cfg, "AUTOLEARN_QUARANTINE_MIN_CONF_PCT", 95.0))
                    for res in results:
                        identity = res.get('identity', 'Desconocido')
                        if identity == "Desconocido":
                            continue
                        confidence = float(res.get('confidence', 0.0))  # 0..100
                        if confidence >= min_conf:
                            bbox = res.get('bbox', None)
                            if bbox is not None:
                                autolearn.maybe_queue(frame, identity, bbox, confidence_pct=confidence)

                    # Tick periódico (cada N frames definidos en config)
                    try:
                        autolearn.maybe_process_periodically(
                            frame_count,
                            every_n=getattr(cfg, "AUTOLEARN_PROCESS_EVERY_N_FRAMES", 30)
                        )
                    except Exception as e:
                        logger.error(f"AutoLearn periodic error: {e}")

            # Update performance metrics
            process_time = performance_monitor.end_timing()
            fps_counter.update(process_time)

            # Display FPS + AutoLearn overlay
            current_fps = fps_counter.get_fps()
            cv2.putText(
                frame, f"FPS: {current_fps:.1f}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                getattr(cfg, "FONT_SCALE", 0.5),
                (255, 255, 255), 1
            )

            if autolearn:
                try:
                    qlen = autolearn.queue_len()
                except Exception:
                    qlen = 0
                draw_autolearn_status(frame, autolearn.enabled, qlen)

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
            elif key == ord('a'):
                if autolearn:
                    autolearn.enabled = not autolearn.enabled
                    logger.info(f"AutoLearn {'habilitado' if autolearn.enabled else 'deshabilitado'}")
                else:
                    logger.info("AutoLearn no disponible")
            elif key == ord('p'):
                if autolearn:
                    try:
                        n_promoted, n_skipped = autolearn.force_process_now()
                        logger.info(f"AutoLearn: procesado inmediato. Promovidos={n_promoted}, Omitidos={n_skipped}")
                    except Exception as e:
                        logger.error(f"AutoLearn force_process_now error: {e}")
                else:
                    logger.info("AutoLearn no disponible")

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

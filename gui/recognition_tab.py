import cv2
import numpy as np
import logging

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

from main.recognition import config as cfg
from main.recognition.camera_manager import CameraManager

logger = logging.getLogger(__name__)

# --- AutoLearn (GUI) ---------------------------------------------------------
try:
    from main.recognition.autolearn import AutoLearner as _AutoLearnerGUI
    _HAS_AUTOLEARN_GUI = True
except Exception as e:
    _AutoLearnerGUI = None
    _HAS_AUTOLEARN_GUI = False
    logger.warning(f"[GUI] AutoLearn no disponible en recognition_tab: {e}")

# --- Performance Monitor (JSONL) ---------------------------------------------
try:
    from main.recognition.perf_monitor import PerformanceMonitor as _PerfMonitor
    _HAS_PERF_MONITOR = True
except Exception as e:
    _PerfMonitor = None
    _HAS_PERF_MONITOR = False
    logger.warning(f"[GUI] PerformanceMonitor no disponible en recognition_tab: {e}")


class RecognitionTab(QWidget):
    def __init__(self, recognizer, face_processor):
        super().__init__()

        self.recognizer = recognizer
        self.face_processor = face_processor

        # Contador de frames para AutoLearn
        self.frame_counter = 0

        # Instancia de AutoLearn (si está disponible y habilitado en config)
        self.autolearn = None
        if _HAS_AUTOLEARN_GUI and bool(getattr(cfg, "AUTOLEARN_ENABLED", False)):
            try:
                self.autolearn = _AutoLearnerGUI()
                # Por defecto lo dejamos habilitado
                self.autolearn.enabled = True
                logger.info("[GUI] AutoLearn inicializado correctamente en RecognitionTab")
            except Exception as e:
                logger.error(f"[GUI] Error inicializando AutoLearn en RecognitionTab: {e}")
                self.autolearn = None
        else:
            logger.info("[GUI] AutoLearn deshabilitado o no disponible en GUI (config o import)")

        # Mosaico 2x3
        self.tile_rows = 2
        self.tile_cols = 3
        self.page_size = self.tile_rows * self.tile_cols

        # ------------------------------------------
        # DETECTAR CÁMARAS AUTOMÁTICAMENTE
        # ------------------------------------------
        self.camera_manager = CameraManager()
        detected_indexes = self.camera_manager.detect_cameras(max_tested=10)

        self.source_cfgs = []
        for idx in detected_indexes:
            self.source_cfgs.append({
                "type": "LOCAL",
                "index": idx,
                "name": f"Cámara {idx}",
                "enabled": True,
            })

        self.total_cams = len(self.source_cfgs)
        self.total_pages = max(1, (self.total_cams + self.page_size - 1) // self.page_size)
        self.current_page = 0

        self.opened_cameras = []

        # ------------------------------------------
        # UI
        # ------------------------------------------
        self.video_label = QLabel(
            "Cargando cámaras..." if self.total_cams > 0 else "No se detectaron cámaras"
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(640, 360)

        btn_prev = QPushButton("<< Página anterior")
        btn_next = QPushButton("Página siguiente >>")

        btn_prev.clicked.connect(self.prev_page)
        btn_next.clicked.connect(self.next_page)

        # Botones AutoLearn
        self.btn_toggle_autolearn = QPushButton("AutoLearn: OFF")
        self.btn_force_autolearn = QPushButton("Procesar cuarentena")

        self.btn_toggle_autolearn.clicked.connect(self.toggle_autolearn)
        self.btn_force_autolearn.clicked.connect(self.force_autolearn)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)
        btn_layout.addWidget(self.btn_toggle_autolearn)
        btn_layout.addWidget(self.btn_force_autolearn)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, stretch=5)
        layout.addLayout(btn_layout)

        # Ajustar estado inicial del botón AutoLearn
        self._refresh_autolearn_button()
        self.setLayout(layout)

        # ------------------------------------------
        # Abrir todas las cámaras detectadas
        # ------------------------------------------
        if self.total_cams > 0:
            self.open_all_cameras()

        # ------------------------------------------
        # Monitor de rendimiento (JSONL)
        # ------------------------------------------
        self.perf_monitor = None
        if _HAS_PERF_MONITOR and bool(getattr(cfg, "PERF_MONITOR_ENABLED", True)):
            try:
                interval = int(getattr(cfg, "PERF_MONITOR_INTERVAL_SEC", 5))
                self.perf_monitor = _PerfMonitor(interval_sec=interval)
            except Exception as e:
                logger.error(f"[GUI] No se pudo iniciar PerformanceMonitor: {e}")
                self.perf_monitor = None

        # Timer (render loop)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(getattr(cfg, "GUI_REFRESH_INTERVAL_MS", 33))

    # -------------------------------------------------------------------------
    #                   ABRIR CÁMARAS
    # -------------------------------------------------------------------------
    def open_all_cameras(self):
        for src in self.source_cfgs:
            cap = None
            ctype = src.get("type", "LOCAL").upper()

            if ctype == "LOCAL":
                cap = cv2.VideoCapture(src["index"], cv2.CAP_DSHOW)

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, getattr(cfg, "CAMERA_WIDTH", 1920))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(cfg, "CAMERA_HEIGHT", 1080))
                cap.set(cv2.CAP_PROP_FPS, getattr(cfg, "CAMERA_FPS", 30))
                if hasattr(cfg, "CAMERA_BUFFER_SIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.CAMERA_BUFFER_SIZE)

            elif ctype == "RTSP":
                cap = cv2.VideoCapture(src.get("rtsp_url", ""))

            if cap and cap.isOpened():
                logger.info(f"[GUI] Cámara abierta: {src['name']}")
                self.opened_cameras.append({"cfg": src, "cap": cap})
            else:
                logger.warning(f"[GUI] No se pudo abrir cámara: {src['name']}")
                self.opened_cameras.append({"cfg": src, "cap": None})

    # -------------------------------------------------------------------------
    #                   PAGINACIÓN
    # -------------------------------------------------------------------------
    def prev_page(self):
        if self.total_pages > 1:
            self.current_page = (self.current_page - 1) % self.total_pages
            logger.info(f"[GUI] Página {self.current_page+1}/{self.total_pages}")

    def next_page(self):
        if self.total_pages > 1:
            self.current_page = (self.current_page + 1) % self.total_pages
            logger.info(f"[GUI] Página {self.current_page+1}/{self.total_pages}")

    # -------------------------------------------------------------------------
    #                   CONTROLES AUTOLEARN
    # -------------------------------------------------------------------------
    def _refresh_autolearn_button(self):
        """Actualiza el texto del botón de AutoLearn según el estado actual."""
        # Si todavía no existe el botón, nada que hacer
        if not hasattr(self, "btn_toggle_autolearn"):
            return
        if self.autolearn and getattr(self.autolearn, "enabled", False):
            self.btn_toggle_autolearn.setText("AutoLearn: ON")
        else:
            self.btn_toggle_autolearn.setText("AutoLearn: OFF")

    def toggle_autolearn(self):
        """Habilita / deshabilita AutoLearn desde la GUI."""
        if not self.autolearn:
            logger.info("[GUI] AutoLearn no está disponible (toggle ignorado)")
            return
        current = bool(getattr(self.autolearn, "enabled", False))
        self.autolearn.enabled = not current
        state = "habilitado" if self.autolearn.enabled else "deshabilitado"
        logger.info(f"[GUI] AutoLearn {state} desde GUI")
        self._refresh_autolearn_button()

    def force_autolearn(self):
        """Fuerza el procesamiento inmediato de la cola de AutoLearn."""
        if not self.autolearn:
            logger.info("[GUI] AutoLearn no está disponible para force_process_now")
            return
        try:
            promoted, skipped = self.autolearn.force_process_now()
            logger.info(
                f"[GUI] AutoLearn force_process_now -> promovidos={promoted}, omitidos={skipped}"
            )
        except Exception as e:
            logger.error(f"[GUI] Error en AutoLearn.force_process_now(): {e}")

    # -------------------------------------------------------------------------
    #                   MOSAICO 2x3
    # -------------------------------------------------------------------------
    def create_mosaic(self, frames, names):
        tile_h, tile_w = 240, 320
        mosaic_h = tile_h * self.tile_rows
        mosaic_w = tile_w * self.tile_cols

        mosaic = np.zeros((mosaic_h, mosaic_w, 3), np.uint8)

        for i in range(len(frames)):
            row = i // self.tile_cols
            col = i % self.tile_cols

            y1 = row * tile_h
            x1 = col * tile_w

            if frames[i] is not None:
                f = cv2.resize(frames[i], (tile_w, tile_h))
            else:
                f = np.zeros((tile_h, tile_w, 3), np.uint8)
                cv2.putText(
                    f,
                    "Sin señal",
                    (10, tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            mosaic[y1:y1 + tile_h, x1:x1 + tile_w] = f

            # Nombre de la cámara
            if i < len(names):
                cv2.putText(
                    mosaic,
                    names[i],
                    (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        return mosaic

    # -------------------------------------------------------------------------
    #                   UPDATE LOOP
    # -------------------------------------------------------------------------
    def update_frame(self):
        if not self.opened_cameras:
            return

        start = self.current_page * self.page_size
        end = min(start + self.page_size, self.total_cams)
        cams_on_page = self.opened_cameras[start:end]

        frames = []
        names = []

        # --- NUEVO: contadores para métricas de rendimiento ---
        interval_known = 0
        interval_unknown = 0

        for cam_info in cams_on_page:
            cfg_cam = cam_info["cfg"]
            cap = cam_info["cap"]
            names.append(cfg_cam["name"])

            if cap is None or not cap.isOpened():
                frames.append(None)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                frames.append(None)
                continue

            results = self.face_processor.process_frame(frame)
            frame = self.face_processor.draw_results(frame, results)

            # --- Conteo de caras conocidas vs desconocidas ---
            for res in results:
                identity = res.get('identity', 'Desconocido')
                if identity == "Desconocido":
                    interval_unknown += 1
                else:
                    interval_known += 1

            # --- AutoLearn: encolar candidatos de alta confianza y procesar periódicamente ---
            if self.autolearn and getattr(self.autolearn, "enabled", False):
                # Contador global de frames para AutoLearn
                self.frame_counter += 1

                try:
                    min_conf = float(getattr(cfg, "AUTOLEARN_QUARANTINE_MIN_CONF_PCT", 95.0))
                except Exception:
                    min_conf = 95.0

                # Encolar sólo identidades conocidas con confianza suficiente
                for res in results:
                    identity = res.get('identity', 'Desconocido')
                    if identity == "Desconocido":
                        continue
                    confidence = float(res.get('confidence', 0.0))  # 0..100 (mismo criterio que en main.py)
                    if confidence >= min_conf:
                        bbox = res.get('bbox', None)
                        if bbox is not None:
                            try:
                                self.autolearn.maybe_queue(frame, identity, bbox, confidence_pct=confidence)
                            except Exception as e:
                                logger.error(f"[GUI] Error en AutoLearn.maybe_queue(): {e}")

                # Tick periódico (cada N frames definidos en config)
                try:
                    every_n = int(getattr(cfg, "AUTOLEARN_PROCESS_EVERY_N_FRAMES", 30))
                    self.autolearn.maybe_process_periodically(self.frame_counter, every_n=every_n)
                except Exception as e:
                    logger.error(f"[GUI] Error en AutoLearn.maybe_process_periodicamente(): {e}")

                # Overlay simple de estado de AutoLearn en el frame
                try:
                    status = "ON" if self.autolearn.enabled else "OFF"
                    qlen = self.autolearn.queue_len()
                    cv2.putText(
                        frame,
                        f"AutoLearn {status} | Q:{qlen}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                except Exception:
                    # No queremos que falle todo el render por un overlay
                    pass

            frames.append(frame)

        if not frames:
            return

        mosaic = self.create_mosaic(frames, names)

        rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        self.video_label.setPixmap(pix)

        # --- NUEVO: tick del monitor de rendimiento (JSONL, sin imprimir en consola) ---
        if self.perf_monitor is not None:
            try:
                self.perf_monitor.tick(
                    num_known=interval_known,
                    num_unknown=interval_unknown,
                    frames=len(cams_on_page),  # 1 frame por cámara en este ciclo
                )
            except Exception as e:
                logger.error(f"[GUI] Error en perf_monitor.tick(): {e}")

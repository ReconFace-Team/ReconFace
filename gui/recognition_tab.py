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

logger = logging.getLogger(__name__)


class RecognitionTab(QWidget):
    def __init__(self, recognizer, face_processor):
        super().__init__()

        self.recognizer = recognizer
        self.face_processor = face_processor

        # === CONFIGURACIÓN DE MOSAICO ===
        self.tile_rows = 2
        self.tile_cols = 3
        self.page_size = self.tile_rows * self.tile_cols

        # === CÁMARAS DEFINIDAS EN CONFIG ===
        self.source_cfgs = [s for s in cfg.GUI_CAMERA_SOURCES if s.get("enabled", True)]
        self.total_cams = len(self.source_cfgs)
        self.total_pages = max(
            1, (self.total_cams + self.page_size - 1) // self.page_size
        )
        self.current_page = 0

        self.opened_cameras = []

        # === UI PRINCIPAL ===
        self.video_label = QLabel("Cargando cámaras...")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Muy importante: que el label escale el contenido y no fuerce el tamaño de la ventana
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        # Un mínimo razonable para que no se colapse
        self.video_label.setMinimumSize(640, 360)

        btn_prev = QPushButton("<< Página anterior")
        btn_next = QPushButton("Página siguiente >>")

        btn_prev.clicked.connect(self.prev_page)
        btn_next.clicked.connect(self.next_page)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)

        layout = QVBoxLayout()
        # El label tiene más “peso” para ocupar espacio, pero sin crecer infinito
        layout.addWidget(self.video_label, stretch=5)
        layout.addLayout(btn_layout, stretch=0)

        self.setLayout(layout)

        # === INICIAR CÁMARAS ===
        self.open_all_cameras()

        # === TIMER (render loop) ===
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(getattr(cfg, "GUI_REFRESH_INTERVAL_MS", 33))

    # -------------------------------------------------------------------------
    #                     GESTIÓN DE CAMARAS
    # -------------------------------------------------------------------------
    def open_all_cameras(self):
        """Abre todas las cámaras habilitadas definidas en config.GUI_CAMERA_SOURCES."""
        for src in self.source_cfgs:
            cap = None
            ctype = src.get("type", "LOCAL").upper()

            if ctype == "LOCAL":
                cap = cv2.VideoCapture(src.get("index", 0))
            elif ctype == "RTSP":
                cap = cv2.VideoCapture(src.get("rtsp_url", ""))

            if cap and cap.isOpened():
                logger.info(f"[GUI] Cámara abierta: {src.get('name', 'cam')}")
                self.opened_cameras.append({"cfg": src, "cap": cap})
            else:
                logger.warning(
                    f"[GUI] No se pudo abrir cámara: {src.get('name', 'cam')}"
                )
                self.opened_cameras.append({"cfg": src, "cap": None})

    def release_cameras(self):
        for c in self.opened_cameras:
            if c["cap"] is not None:
                c["cap"].release()
        self.opened_cameras.clear()

    # -------------------------------------------------------------------------
    #                          PAGINACIÓN
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
    #                       MOSAICO 2×3 FIJO (RESPONSIVO VIA QLabel)
    # -------------------------------------------------------------------------
    def create_mosaic(self, frames, names):
        """
        Crea un mosaico 2×3 de tamaño fijo (por ejemplo 960x480).
        El QLabel se encarga de escalarlo al tamaño de la ventana.
        """
        tile_h, tile_w = 240, 320   # 3 columnas * 320 = 960; 2 filas * 240 = 480
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
                mosaic[y1:y1 + tile_h, x1:x1 + tile_w] = f
            else:
                # Fondo gris para cámaras no disponibles
                mosaic[y1:y1 + tile_h, x1:x1 + tile_w] = (40, 40, 40)

            # Nombre de la cámara
            cv2.putText(
                mosaic,
                names[i],
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return mosaic

    # -------------------------------------------------------------------------
    #                           UPDATE LOOP
    # -------------------------------------------------------------------------
    def update_frame(self):
        """Lee las cámaras de la página actual, procesa y actualiza el mosaico."""
        if not self.opened_cameras:
            return

        start = self.current_page * self.page_size
        end = min(start + self.page_size, self.total_cams)
        cams_on_page = self.opened_cameras[start:end]

        frames = []
        names = []

        for cam_info in cams_on_page:
            cfg_cam = cam_info["cfg"]
            cap = cam_info["cap"]
            names.append(cfg_cam.get("name", "Cam"))

            if cap is None or not cap.isOpened():
                frames.append(None)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                frames.append(None)
                continue

            # Procesar reconocimiento
            results = self.face_processor.process_frame(frame)
            frame = self.face_processor.draw_results(frame, results)

            frames.append(frame)

        if not frames:
            return

        mosaic = self.create_mosaic(frames, names)

        # Convertir a QImage (el QLabel hará el escalado interno, sin cambiar el tamaño de la ventana)
        rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        self.video_label.setPixmap(pix)

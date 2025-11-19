import time
import threading
import queue
import cv2
from PySide6 import QtCore, QtWidgets

from main.recognition import config as cfg
from main.recognition.camera_manager import CameraManager
from main.recognition.face_recognizer import OptimizedFaceRecognizer
from main.recognition.face_processor import FaceProcessor
from .camera_widget import CameraWidget
from .storage import get_person_status

class _CamWorker(QtCore.QObject):
    """
    Hilo por cámara: captura, procesa cada N frames y emite frame anotado.
    Comparte un único recognizer/face_processor (inyectados desde el panel).
    """
    frameReady = QtCore.Signal(int, object)   # idx_cam, frame_bgr
    stopped    = QtCore.Signal(int)

    def __init__(self, cam_idx, source, recognizer: OptimizedFaceRecognizer, face_processor: FaceProcessor, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.source = source
        self.recognizer = recognizer
        self.face_processor = face_processor
        self._running = False
        self._proc_every = int(getattr(cfg, "GUI_PROCESS_EVERY_N_FRAMES", 2))
        self._frame_count = 0

    @QtCore.Slot()
    def start(self):
        self._running = True
        cap = None
        try:
            # Usa CameraManager internamente para RTSP, etc., pero aquí abrimos cv2 directo por simplicidad
            cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF if isinstance(self.source, int) else cv2.CAP_FFMPEG)
            if isinstance(self.source, int):
                # Opcionales: set de ancho/alto
                if getattr(cfg, "CAMERA_WIDTH", None):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
                if getattr(cfg, "CAMERA_HEIGHT", None):
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
            if not cap.isOpened():
                self.stopped.emit(self.cam_idx)
                return

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                self._frame_count += 1
                annotated = frame
                if (self._frame_count % self._proc_every) == 0:
                    # Reconoce + dibuja
                    results = self.face_processor.process_frame(frame)
                    annotated = self._draw_access_policy(self.face_processor.draw_results(frame.copy(), results), results)

                self.frameReady.emit(self.cam_idx, annotated)

            cap.release()
            self.stopped.emit(self.cam_idx)
        except Exception:
            if cap is not None:
                cap.release()
            self.stopped.emit(self.cam_idx)

    def stop(self):
        self._running = False

    def _draw_access_policy(self, frame, results):
        """
        Pinta borde/label distinto si persona está en blacklist/whitelist.
        No bloquea la detección; solo visual y útil para alarmística futura.
        """
        for r in results:
            name = r.get("identity", "Desconocido")
            bbox = r.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            status = get_person_status(name)
            if status == "black":
                color = (0, 0, 255)   # Rojo fuerte
                label = f"{name} (BLACK)"
            elif status == "white":
                color = (0, 255, 0)   # Verde
                label = f"{name} (WHITE)"
            else:
                color = (255, 255, 255)
                label = name
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if getattr(cfg, "GUI_DRAW_NAMES", True):
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


class RecognitionPanel(QtWidgets.QWidget):
    """
    Panel con grilla de cámaras y pipeline de reconocimiento integrado.
    Crea UNA instancia de recognizer/face_processor y la comparte a todos los hilos.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._workers = []
        self._threads = []

        # Título + botones
        header = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Iniciar")
        self.btn_stop  = QtWidgets.QPushButton("Detener")
        header.addWidget(QtWidgets.QLabel("Reconocimiento (multi-cámara)"))
        header.addStretch(1)
        header.addWidget(self.btn_start)
        header.addWidget(self.btn_stop)

        # Grilla de cámaras (hasta 4)
        self._grid = QtWidgets.QGridLayout()
        self._cams = []
        max_cams = min(len(getattr(cfg, "CAMERA_SOURCES", [0])), int(getattr(cfg, "GUI_MAX_CAMERAS", 4)))
        rows = 2
        cols = 2
        for i in range(max_cams):
            w = CameraWidget(title=f"Cámara {i+1}")
            self._cams.append(w)
            self._grid.addWidget(w, i // cols, i % cols)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(header)
        lay.addLayout(self._grid, 1)

        self.btn_start.clicked.connect(self.start_all)
        self.btn_stop.clicked.connect(self.stop_all)

        # Crea objetos de reconocimiento compartidos
        self._recognizer = OptimizedFaceRecognizer()
        self._face_processor = FaceProcessor(self._recognizer)

    def start_all(self):
        self.stop_all()
        sources = getattr(cfg, "CAMERA_SOURCES", [0])
        for idx, src in enumerate(sources[:len(self._cams)]):
            t = QtCore.QThread(self)
            w = _CamWorker(idx, src, self._recognizer, self._face_processor)
            w.moveToThread(t)
            w.frameReady.connect(self._on_frame)
            w.stopped.connect(self._on_worker_stopped)
            t.started.connect(w.start)
            self._workers.append(w)
            self._threads.append(t)
            t.start()

    def stop_all(self):
        for w in self._workers:
            try:
                w.stop()
            except Exception:
                pass
        for t in self._threads:
            t.quit()
            t.wait(1000)
        self._workers.clear()
        self._threads.clear()

    @QtCore.Slot(int, object)
    def _on_frame(self, cam_idx, frame_bgr):
        if 0 <= cam_idx < len(self._cams):
            self._cams[cam_idx].show_frame(frame_bgr)

    @QtCore.Slot(int)
    def _on_worker_stopped(self, cam_idx):
        # Podrías informar en UI si alguna cámara cae
        pass

    def closeEvent(self, e):
        self.stop_all()
        return super().closeEvent(e)

# gui/recognition_panel.py

import time
import cv2
from PySide6 import QtCore, QtWidgets

# === MÓDULO DE RECONOCIMIENTO (ruta correcta según tu estructura) ===
from main import config as cfg
from main.recognition.face_recognizer import OptimizedFaceRecognizer
from main.recognition.face_processor import FaceProcessor

from .camera_widget import CameraWidget
from .storage import get_person_status

# === AUTOLEARN ===
try:
    from main.recognition.autolearn import AutoLearner
    _HAS_AUTOLEARN = True
except Exception as e:
    print("[GUI] No se pudo importar AutoLearner:", e)
    AutoLearner = None
    _HAS_AUTOLEARN = False


class _CamWorker(QtCore.QObject):
    """
    Worker por cámara: captura -> procesa -> encola AutoLearn -> emite a la GUI.
    """
    frameReady = QtCore.Signal(int, object)   # (cam_idx, frame_bgr)
    stopped = QtCore.Signal(int)             # cam_idx

    def _init_(self, cam_idx, source, recognizer, face_processor, autolearn=None, parent=None):
        super()._init_(parent)
        self.cam_idx = cam_idx
        self.source = source
        self.recognizer = recognizer
        self.face_processor = face_processor
        self.autolearn = autolearn
        self._running = False
        self._frame_count = 0
        # procesar solo cada N frames (para no matar la CPU)
        self._proc_every = int(getattr(cfg, "GUI_PROCESS_EVERY_N_FRAMES", 2))

    @QtCore.Slot()
    def start(self):
        self._running = True
        cap = None
        try:
            is_local = isinstance(self.source, int)
            backend = cv2.CAP_MSMF if is_local else cv2.CAP_FFMPEG

            cap = cv2.VideoCapture(self.source, backend)

            if is_local:
                if getattr(cfg, "CAMERA_WIDTH", None):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
                if getattr(cfg, "CAMERA_HEIGHT", None):
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)

            if not cap.isOpened():
                print(f"[CamWorker {self.cam_idx}] No se pudo abrir la fuente {self.source}")
                self.stopped.emit(self.cam_idx)
                return

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                self._frame_count += 1
                annotated = frame

                # === PROCESAR SOLO CADA N FRAMES ===
                if (self._frame_count % self._proc_every) == 0:
                    results = self.face_processor.process_frame(frame)
                    annotated = self._draw_access_status(
                        self.face_processor.draw_results(frame.copy(), results),
                        results
                    )

                    # === AUTOLEARN: ENCOLAR FRAMES DE ALTA CONFIANZA ===
                    if self.autolearn is not None and self.autolearn.enabled:
                        try:
                            min_conf = float(
                                getattr(cfg, "AUTOLEARN_QUARANTINE_MIN_CONF_PCT", 95.0)
                            )
                        except Exception:
                            min_conf = 95.0

                        for r in results:
                            name = r.get("identity", "Desconocido")
                            if name == "Desconocido":
                                continue

                            conf = float(r.get("confidence", 0.0))  # 0..100
                            if conf < min_conf:
                                continue

                            bbox = r.get("bbox")
                            if bbox is None:
                                continue

                            # DEBUG visible en consola:
                            print(f"[CamWorker {self.cam_idx}] AutoLearn enqueue {name} conf={conf:.1f}")
                            # Encolar en cuarentena
                            self.autolearn.maybe_queue(frame, name, bbox, confidence_pct=conf)

                        # Tick periódico (igual que en main.py de consola)
                        try:
                            self.autolearn.maybe_process_periodically(
                                self._frame_count,
                                every_n=getattr(cfg, "AUTOLEARN_PROCESS_EVERY_N_FRAMES", 30)
                            )
                        except Exception as e:
                            print(f"[CamWorker {self.cam_idx}] AutoLearn periodic error:", e)

                # Enviar frame a la GUI
                self.frameReady.emit(self.cam_idx, annotated)

            cap.release()
            self.stopped.emit(self.cam_idx)

        except Exception as e:
            print(f"[CamWorker {self.cam_idx}] ERROR:", e)
            if cap is not None:
                cap.release()
            self.stopped.emit(self.cam_idx)

    def stop(self):
        self._running = False

    # ----- Dibuja colores según white/black list -----
    def _draw_access_status(self, frame, results):
        for r in results:
            name = r.get("identity", "Desconocido")
            bbox = r.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            status = get_person_status(name)  # "white", "black", "unknown"

            if status == "black":
                color = (0, 0, 255)
                label = f"{name} (BLACK)"
            elif status == "white":
                color = (0, 255, 0)
                label = f"{name} (WHITE)"
            else:
                color = (255, 255, 255)
                label = name

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


class RecognitionPanel(QtWidgets.QWidget):
    """
    Panel principal de reconocimiento (multi-cámara) + AutoLearn.
    """
    def _init_(self, parent=None):
        super()._init_(parent)
        self._workers = []
        self._threads = []
        self._autolearn = None

        # --------- CABECERA / CONTROLES ---------
        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Reconocimiento Facial (Multi-cámara)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.lbl_autolearn = QtWidgets.QLabel("AutoLearn: OFF (cola: 0)")
        self.btn_toggle_autolearn = QtWidgets.QPushButton("AutoLearn ON/OFF")
        self.btn_force_al = QtWidgets.QPushButton("Procesar AutoLearn ahora")
        self.btn_start = QtWidgets.QPushButton("Iniciar cámaras")
        self.btn_stop = QtWidgets.QPushButton("Detener cámaras")

        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.lbl_autolearn)
        header_layout.addWidget(self.btn_toggle_autolearn)
        header_layout.addWidget(self.btn_force_al)
        header_layout.addWidget(self.btn_start)
        header_layout.addWidget(self.btn_stop)

        # --------- GRID DE CÁMARAS ---------
        grid = QtWidgets.QGridLayout()
        self._cams = []

        sources = getattr(cfg, "CAMERA_SOURCES", [0])  # si no existe, una sola cámara local
        max_cams = min(len(sources), getattr(cfg, "GUI_MAX_CAMERAS", 4))

        for i in range(max_cams):
            w = CameraWidget(title=f"Cámara {i+1}")
            self._cams.append(w)
            grid.addWidget(w, i // 2, i % 2)

        # Layout general
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header_layout)
        layout.addLayout(grid)

        # --------- LÓGICA PRINCIPAL ---------
        self.btn_start.clicked.connect(self.start_all)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_toggle_autolearn.clicked.connect(self.toggle_autolearn)
        self.btn_force_al.clicked.connect(self.force_autolearn_now)

        # Reconocedor compartido
        self._recognizer = OptimizedFaceRecognizer()
        self._face_processor = FaceProcessor(self._recognizer)

        # AutoLearn global
        if _HAS_AUTOLEARN and getattr(cfg, "AUTOLEARN_ENABLED", True):
            try:
                self._autolearn = AutoLearner()
                self._autolearn.enabled = True
                self._update_autolearn_label()
            except Exception as e:
                print("[GUI] No se pudo inicializar AutoLearner:", e)
                self._autolearn = None
                self.lbl_autolearn.setText("AutoLearn: ERROR")
        else:
            self.lbl_autolearn.setText("AutoLearn: OFF (deshabilitado)")

    # ====== CONTROL DE CÁMARAS ======
    def start_all(self):
        self.stop_all()
        sources = getattr(cfg, "CAMERA_SOURCES", [0])

        for idx, src in enumerate(sources[:len(self._cams)]):
            t = QtCore.QThread(self)
            w = _CamWorker(idx, src, self._recognizer, self._face_processor, self._autolearn)

            w.moveToThread(t)
            w.frameReady.connect(self._on_frame)
            w.stopped.connect(self._on_worker_stopped)
            t.started.connect(w.start)

            self._workers.append(w)
            self._threads.append(t)
            t.start()

    def stop_all(self):
        # Detener workers
        for w in self._workers:
            try:
                w.stop()
            except Exception:
                pass

        # Detener threads
        for t in self._threads:
            t.quit()
            t.wait(500)

        self._workers.clear()
        self._threads.clear()

    @QtCore.Slot(int, object)
    def _on_frame(self, cam_idx, frame_bgr):
        if 0 <= cam_idx < len(self._cams):
            self._cams[cam_idx].show_frame(frame_bgr)
        self._update_autolearn_label()

    @QtCore.Slot(int)
    def _on_worker_stopped(self, idx):
        # aquí podrías mostrar algo si una cámara se cae
        pass

    # ====== AUTOLEARN CONTROLES GUI ======
    def _update_autolearn_label(self):
        if self._autolearn is None:
            self.lbl_autolearn.setText("AutoLearn: OFF")
            return
        try:
            qlen = self._autolearn.queue_len()
        except Exception:
            qlen = 0
        status = "ON" if self._autolearn.enabled else "OFF"
        self.lbl_autolearn.setText(f"AutoLearn: {status} (cola: {qlen})")

    def toggle_autolearn(self):
        if self._autolearn is None:
            QtWidgets.QMessageBox.information(
                self, "AutoLearn", "AutoLearn no está disponible o falló al inicializar."
            )
            return
        self._autolearn.enabled = not self._autolearn.enabled
        self._update_autolearn_label()

    def force_autolearn_now(self):
        """
        Equivalente a la tecla 'p' en tu main de consola:
        procesa inmediatamente la cuarentena.
        """
        if self._autolearn is None:
            QtWidgets.QMessageBox.information(
                self, "AutoLearn", "AutoLearn no está disponible."
            )
            return
        try:
            n_promoted, n_skipped = self._autolearn.force_process_now()
            self._update_autolearn_label()
            QtWidgets.QMessageBox.information(
                self,
                "AutoLearn",
                f"Procesado de cuarentena terminado.\n"
                f"Promovidos: {n_promoted}\n"
                f"Omitidos: {n_skipped}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "AutoLearn",
                f"Error al procesar cuarentena:\n{e}"
            )

    def closeEvent(self, e):
        self.stop_all()
        super().closeEvent(e)
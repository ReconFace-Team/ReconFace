import sys
import subprocess
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from main.recognition import config as cfg


class TrainingPanel(QtWidgets.QWidget):
    """
    Lanza el pipeline de entrenamiento (embeddings) ejecutando el módulo training.
    Captura la salida en una consola de texto.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.folder_edit = QtWidgets.QLineEdit(str(Path(cfg.INPUT_DIR)))
        self.btn_browse  = QtWidgets.QPushButton("Elegir carpeta de imágenes…")
        self.btn_run     = QtWidgets.QPushButton("Ejecutar entrenamiento")
        self.console     = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Carpeta INPUT_DIR:"))
        top.addWidget(self.folder_edit, 1)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_run)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.console, 1)

        self._proc = None
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_run.clicked.connect(self._on_run)

    def _on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecciona carpeta de imágenes", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d)

    def append_console(self, text):
        self.console.appendPlainText(text)

    def _on_run(self):
        if self._proc is not None:
            self.append_console("⚠️ Ya hay un proceso en ejecución.")
            return

        # Ajusta cfg.INPUT_DIR en tiempo de ejecución (opcional)
        # Recomendado: dejar config.py tal cual y organizar la carpeta de imágenes.
        self.append_console("🚀 Lanzando entrenamiento: python -m main.training.main")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "main.training.main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[1])  # raíz del repo (donde está main/)
        )

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll_output)
        self._timer.start(100)

    def _poll_output(self):
        if self._proc is None:
            return
        if self._proc.stdout is not None:
            try:
                line = self._proc.stdout.readline()
            except Exception:
                line = b""
            if line:
                try:
                    text = line.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    text = str(line)
                self.append_console(text)
            if self._proc.poll() is not None and not line:
                # finalizó
                rc = self._proc.returncode
                self.append_console(f"✅ Entrenamiento finalizado (rc={rc})")
                self._timer.stop()
                self._proc = None

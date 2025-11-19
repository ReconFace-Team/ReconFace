"""
Main GUI application for the face recognition system
- Orquesta las pestañas:
  * Reconocimiento (multi-cámara + AutoLearn)
  * Listas (whitelist / blacklist)
  * Entrenamiento (gestión de imágenes y embeddings)
"""

import sys
import logging

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
import cv2

from main.recognition import config as cfg
from main.recognition.face_recognizer import OptimizedFaceRecognizer
from main.recognition.face_processor import FaceProcessor
from main.recognition.utils import (
    setup_logging,
    validate_directories,
)

from .recognition_tab import RecognitionTab
from .lists_tab import ListsTab
from .training_tab import TrainingTab


logger = setup_logging()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            getattr(cfg, "GUI_WINDOW_TITLE", "Sistema de Reconocimiento Facial - GUI")
        )
        self.resize(1280, 720)

        # Validar recursos base (embeddings, modelos...)
        if not validate_directories():
            QMessageBox.critical(
                self, "Error", "Validación de directorios falló. Revisa los logs."
            )
            raise RuntimeError("validate_directories() failed")

        # Inicializar núcleo de reconocimiento
        logger.info("Inicializando núcleo de reconocimiento para la GUI...")
        self.recognizer = OptimizedFaceRecognizer()
        if len(self.recognizer.face_embeddings) == 0:
            QMessageBox.critical(
                self, "Error", "No se encontraron embeddings válidos."
            )
            raise RuntimeError("No embeddings loaded")

        self.face_processor = FaceProcessor(self.recognizer)

        # Crear pestañas
        tabs = QTabWidget()

        self.recognition_tab = RecognitionTab(self.recognizer, self.face_processor)
        self.lists_tab = ListsTab()
        self.training_tab = TrainingTab()

        tabs.addTab(self.recognition_tab, "Reconocimiento")
        tabs.addTab(self.lists_tab, "Listas (White/Black)")
        tabs.addTab(self.training_tab, "Entrenamiento")

        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        """Al cerrar la ventana, liberar recursos."""
        try:
            self.recognition_tab.release_cameras()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

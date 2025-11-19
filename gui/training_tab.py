import os
import sys
import shutil
import subprocess
import logging

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)

from main.recognition import config as cfg

logger = logging.getLogger(__name__)


class TrainingTab(QWidget):
    """
    Tab para manejar imágenes de entrenamiento:
    - Lista de personas (carpetas en TRAINING_IMAGES_DIR)
    - Añadir/eliminar personas
    - Añadir/eliminar imágenes
    - Lanzar script de entrenamiento (TRAINING_SCRIPT_PATH)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_dir = cfg.TRAINING_IMAGES_DIR
        self.training_script = cfg.TRAINING_SCRIPT_PATH

        os.makedirs(self.training_dir, exist_ok=True)

        self._build_ui()
        self.load_persons()

    # ======================= UI =======================

    def _build_ui(self):
        main_layout = QHBoxLayout()

        # Lado izquierdo: personas
        persons_group = QGroupBox("Personas (carpetas de entrenamiento)")
        persons_layout = QVBoxLayout()
        self.person_list = QListWidget()

        btn_persons_layout = QHBoxLayout()
        self.btn_refresh_persons = QPushButton("Refrescar")
        self.btn_add_person = QPushButton("Nueva persona")
        self.btn_del_person = QPushButton("Eliminar persona")
        btn_persons_layout.addWidget(self.btn_refresh_persons)
        btn_persons_layout.addWidget(self.btn_add_person)
        btn_persons_layout.addWidget(self.btn_del_person)

        persons_layout.addWidget(self.person_list)
        persons_layout.addLayout(btn_persons_layout)
        persons_group.setLayout(persons_layout)

        # Lado derecho: imágenes de la persona
        images_group = QGroupBox("Imágenes de la persona seleccionada")
        images_layout = QVBoxLayout()
        self.images_list = QListWidget()

        btn_images_layout = QHBoxLayout()
        self.btn_add_images = QPushButton("Añadir imágenes...")
        self.btn_del_image = QPushButton("Eliminar imagen")
        self.btn_open_folder = QPushButton("Abrir carpeta")
        btn_images_layout.addWidget(self.btn_add_images)
        btn_images_layout.addWidget(self.btn_del_image)
        btn_images_layout.addWidget(self.btn_open_folder)

        images_layout.addWidget(self.images_list)
        images_layout.addLayout(btn_images_layout)
        images_group.setLayout(images_layout)

        # Abajo: botón de entrenamiento
        train_group = QGroupBox("Entrenamiento")
        train_layout = QHBoxLayout()
        self.btn_run_training = QPushButton("Ejecutar entrenamiento ahora")
        train_layout.addWidget(self.btn_run_training)
        train_group.setLayout(train_layout)

        # Componer layout principal
        main_layout.addWidget(persons_group, 1)
        main_layout.addWidget(images_group, 2)

        outer_layout = QVBoxLayout()
        outer_layout.addLayout(main_layout)
        outer_layout.addWidget(train_group)

        self.setLayout(outer_layout)

        # Conectar señales
        self.person_list.currentItemChanged.connect(self.on_person_selected)
        self.btn_refresh_persons.clicked.connect(self.load_persons)
        self.btn_add_person.clicked.connect(self.add_person)
        self.btn_del_person.clicked.connect(self.delete_person)

        self.btn_add_images.clicked.connect(self.add_images_to_person)
        self.btn_del_image.clicked.connect(self.delete_selected_image)
        self.btn_open_folder.clicked.connect(self.open_person_folder)

        self.btn_run_training.clicked.connect(self.run_training_script)

    # ======================= Personas =======================

    def load_persons(self):
        self.person_list.clear()
        try:
            for name in sorted(os.listdir(self.training_dir)):
                full_path = os.path.join(self.training_dir, name)
                if os.path.isdir(full_path):
                    self.person_list.addItem(QListWidgetItem(name))
        except Exception as e:
            logger.error(f"Error listando personas en {self.training_dir}: {e}")

        self.images_list.clear()

    def add_person(self):
        name, ok = QInputDialog.getText(
            self, "Nueva persona", "Nombre de la persona:"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        new_dir = os.path.join(self.training_dir, name)
        if os.path.exists(new_dir):
            QMessageBox.warning(
                self, "Persona existente", "Ya existe una carpeta con ese nombre."
            )
            return
        try:
            os.makedirs(new_dir, exist_ok=False)
            logger.info(f"Carpeta de entrenamiento creada: {new_dir}")
            self.load_persons()
        except Exception as e:
            logger.error(f"No se pudo crear carpeta de persona: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo crear la carpeta: {e}")

    def delete_person(self):
        item = self.person_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "Eliminar persona", "Selecciona una persona primero."
            )
            return
        name = item.text()
        person_dir = os.path.join(self.training_dir, name)

        # Solo permitimos eliminar si está vacía
        if os.listdir(person_dir):
            QMessageBox.warning(
                self,
                "Carpeta no vacía",
                "La carpeta no está vacía. Elimina las imágenes primero.",
            )
            return

        resp = QMessageBox.question(
            self,
            "Eliminar persona",
            f"¿Seguro que quieres eliminar la carpeta de '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        try:
            os.rmdir(person_dir)
            logger.info(f"Carpeta de persona eliminada: {person_dir}")
            self.load_persons()
        except Exception as e:
            logger.error(f"No se pudo eliminar carpeta de persona: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo eliminar la carpeta: {e}")

    # ======================= Imágenes =======================

    def current_person_dir(self):
        item = self.person_list.currentItem()
        if not item:
            return None
        name = item.text()
        return os.path.join(self.training_dir, name)

    def on_person_selected(self, current, previous):
        self.load_images_for_person()

    def load_images_for_person(self):
        self.images_list.clear()
        person_dir = self.current_person_dir()
        if not person_dir:
            return
        try:
            for fname in sorted(os.listdir(person_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images_list.addItem(QListWidgetItem(fname))
        except Exception as e:
            logger.error(f"Error listando imágenes en {person_dir}: {e}")

    def add_images_to_person(self):
        person_dir = self.current_person_dir()
        if not person_dir:
            QMessageBox.information(
                self, "Añadir imágenes", "Selecciona una persona primero."
            )
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imágenes",
            "",
            "Imágenes (*.jpg *.jpeg *.png)",
        )
        if not files:
            return

        for src in files:
            try:
                base = os.path.basename(src)
                dst = os.path.join(person_dir, base)
                if os.path.exists(dst):
                    # renombrar si ya existe
                    name, ext = os.path.splitext(base)
                    i = 1
                    while True:
                        new_name = f"{name}_{i}{ext}"
                        dst = os.path.join(person_dir, new_name)
                        if not os.path.exists(dst):
                            break
                        i += 1
                shutil.copy2(src, dst)
                logger.info(f"Copiada imagen de entrenamiento: {dst}")
            except Exception as e:
                logger.error(f"Error copiando imagen {src}: {e}")

        self.load_images_for_person()

    def delete_selected_image(self):
        person_dir = self.current_person_dir()
        if not person_dir:
            QMessageBox.information(
                self, "Eliminar imagen", "Selecciona una persona primero."
            )
            return

        item = self.images_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "Eliminar imagen", "Selecciona una imagen de la lista."
            )
            return

        fname = item.text()
        full_path = os.path.join(person_dir, fname)

        resp = QMessageBox.question(
            self,
            "Eliminar imagen",
            f"¿Seguro que quieres eliminar '{fname}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        try:
            os.remove(full_path)
            logger.info(f"Imagen de entrenamiento eliminada: {full_path}")
            self.load_images_for_person()
        except Exception as e:
            logger.error(f"No se pudo eliminar la imagen: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo eliminar la imagen: {e}")

    def open_person_folder(self):
        person_dir = self.current_person_dir()
        if not person_dir:
            QMessageBox.information(
                self, "Abrir carpeta", "Selecciona una persona primero."
            )
            return

        # Abrir en explorador de Windows
        try:
            os.startfile(person_dir)
        except Exception as e:
            logger.error(f"No se pudo abrir la carpeta: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo abrir la carpeta: {e}")

    # ======================= Entrenamiento =======================

    def run_training_script(self):
        if not os.path.exists(self.training_script):
            QMessageBox.critical(
                self,
                "Script no encontrado",
                f"No se encontró el script de entrenamiento:\n{self.training_script}",
            )
            return

        resp = QMessageBox.question(
            self,
            "Ejecutar entrenamiento",
            "¿Quieres lanzar el script de entrenamiento ahora?\n"
            "Se ejecutará en un proceso aparte con este mismo intérprete de Python.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        try:
            # Ejecutamos en segundo plano con el mismo intérprete
            subprocess.Popen(
                [sys.executable, self.training_script],
                cwd=os.path.dirname(self.training_script),
            )
            QMessageBox.information(
                self,
                "Entrenamiento lanzado",
                "El script de entrenamiento se ha lanzado en segundo plano.\n"
                "Revisa la consola / logs del script para ver el progreso.",
            )
            logger.info(f"Entrenamiento lanzado: {self.training_script}")
        except Exception as e:
            logger.error(f"No se pudo lanzar el script de entrenamiento: {e}")
            QMessageBox.critical(
                self, "Error", f"No se pudo lanzar el entrenamiento: {e}"
            )

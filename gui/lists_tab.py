import os
import json
import logging

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QMessageBox,
)

from main.recognition import config as cfg

logger = logging.getLogger(__name__)


class ListsTab(QWidget):
    """Tab para gestionar whitelist y blacklist."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.whitelist_path = cfg.WHITELIST_PATH
        self.blacklist_path = cfg.BLACKLIST_PATH

        self._ensure_files()
        self._build_ui()
        self.load_lists()

    # ======================= Archivos =======================

    def _ensure_files(self):
        """Crea archivos JSON vacíos si no existen."""
        os.makedirs(os.path.dirname(self.whitelist_path), exist_ok=True)
        for path in [self.whitelist_path, self.blacklist_path]:
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)

    # ======================= UI =======================

    def _build_ui(self):
        layout = QHBoxLayout()

        # Whitelist
        wl_group = QGroupBox("Whitelist (permitidos)")
        wl_layout = QVBoxLayout()
        self.wl_list = QListWidget()
        wl_layout.addWidget(self.wl_list)
        wl_group.setLayout(wl_layout)

        # Botones centrales
        btn_layout = QVBoxLayout()
        self.btn_to_black = QPushButton(">> A Blacklist")
        self.btn_to_white = QPushButton("<< A Whitelist")
        self.btn_save = QPushButton("Guardar cambios")
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_to_black)
        btn_layout.addWidget(self.btn_to_white)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)

        # Blacklist
        bl_group = QGroupBox("Blacklist (bloqueados)")
        bl_layout = QVBoxLayout()
        self.bl_list = QListWidget()
        bl_layout.addWidget(self.bl_list)
        bl_group.setLayout(bl_layout)

        layout.addWidget(wl_group)
        layout.addLayout(btn_layout)
        layout.addWidget(bl_group)

        self.setLayout(layout)

        # Conectar señales
        self.btn_to_black.clicked.connect(self.move_to_blacklist)
        self.btn_to_white.clicked.connect(self.move_to_whitelist)
        self.btn_save.clicked.connect(self.save_lists)

    # ======================= Carga / helpers =======================

    def load_lists(self):
        """Carga las listas desde JSON."""
        try:
            with open(self.whitelist_path, "r", encoding="utf-8") as f:
                wl = json.load(f)
        except Exception:
            wl = []
        try:
            with open(self.blacklist_path, "r", encoding="utf-8") as f:
                bl = json.load(f)
        except Exception:
            bl = []

        self.wl_list.clear()
        for name in wl:
            self.wl_list.addItem(QListWidgetItem(name))

        self.bl_list.clear()
        for name in bl:
            self.bl_list.addItem(QListWidgetItem(name))

    def _get_list_items(self, qlistwidget):
        """Devuelve una lista de strings con los items en un QListWidget."""
        items = []
        for i in range(qlistwidget.count()):
            items.append(qlistwidget.item(i).text())
        return items

    # ======================= Movimientos =======================

    def move_to_blacklist(self):
        selected = self.wl_list.selectedItems()
        for item in selected:
            name = item.text()
            self.wl_list.takeItem(self.wl_list.row(item))
            self.bl_list.addItem(QListWidgetItem(name))

    def move_to_whitelist(self):
        selected = self.bl_list.selectedItems()
        for item in selected:
            name = item.text()
            self.bl_list.takeItem(self.bl_list.row(item))
            self.wl_list.addItem(QListWidgetItem(name))

    # ======================= Guardar =======================

    def save_lists(self):
        wl = self._get_list_items(self.wl_list)
        bl = self._get_list_items(self.bl_list)
        try:
            with open(self.whitelist_path, "w", encoding="utf-8") as f:
                json.dump(wl, f, ensure_ascii=False, indent=2)
            with open(self.blacklist_path, "w", encoding="utf-8") as f:
                json.dump(bl, f, ensure_ascii=False, indent=2)
            QMessageBox.information(
                self, "Guardar listas", "Listas guardadas correctamente."
            )
            logger.info(f"Whitelist guardada: {wl}")
            logger.info(f"Blacklist guardada: {bl}")
        except Exception as e:
            logger.error(f"Error guardando listas: {e}")
            QMessageBox.critical(
                self, "Error", f"No se pudieron guardar las listas: {e}"
            )

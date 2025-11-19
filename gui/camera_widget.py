import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from main.recognition import config as cfg



class CameraWidget(QtWidgets.QFrame):
    """
    Widget simple que pinta frames (numpy BGR) en un QLabel.
    No abre la cámara él mismo: recibe frames desde afuera (panel de reconocimiento).
    """
    def __init__(self, title="Cámara", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._label = QtWidgets.QLabel(self)
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._title = QtWidgets.QLabel(title, self)
        self._title.setStyleSheet("font-weight: 600;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._title)
        lay.addWidget(self._label, 1)

    def set_title(self, text: str):
        self._title.setText(text)

    def show_frame(self, frame_bgr):
        if frame_bgr is None:
            return
        # BGR -> RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self._label.setPixmap(pix)

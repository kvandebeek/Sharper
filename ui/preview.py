import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import Qt

class PreviewWidget(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())

        # Correct render hints for Qt6
        self.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform
        )

    def update_image(self, img):
        if img is None:
            return

        arr = np.array(img)

        # Drop alpha if present
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]

        # If grayscale → expand to RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # Normalize floats
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, c * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        # Update scene
        self.scene().clear()
        self.scene().addPixmap(pix)

        # Fit image to view
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

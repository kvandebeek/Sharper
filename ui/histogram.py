from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF
from PySide6.QtCore import Qt, QPointF
import numpy as np


class HistogramWidget(QWidget):
    def __init__(self, parent=None, threshold=0.02):
        super().__init__(parent)
        self.hist_r = None
        self.hist_g = None
        self.hist_b = None
        self.threshold = threshold

    def update_image(self, img):
        if (
            img is None
            or img.ndim != 3
            or img.shape[2] != 3
        ):
            return

        mask = img > self.threshold
        mask_any = np.any(mask, axis=2)

        if not np.any(mask_any):
            self.hist_r = None
            self.update()
            return

        sig = img[mask_any]
        sig_u8 = (sig * 255).astype(np.uint8)

        r = sig_u8[:, 0]
        g = sig_u8[:, 1]
        b = sig_u8[:, 2]

        self.hist_r, _ = np.histogram(r, 256, (0,255))
        self.hist_g, _ = np.histogram(g, 256, (0,255))
        self.hist_b, _ = np.histogram(b, 256, (0,255))

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        if self.hist_r is None:
            return

        w = self.width()
        h = self.height()

        # independent vertical scale for each channel
        max_r = max(1, int(self.hist_r.max()))
        max_g = max(1, int(self.hist_g.max()))
        max_b = max(1, int(self.hist_b.max()))

        def draw_curve(hist, color, max_v):
            pen = QPen(QColor(*color), 2)
            painter.setPen(pen)

            poly = QPolygonF()
            for i in range(256):
                x = i / 255.0 * w
                y = h - (hist[i] / max_v) * h
                poly.append(QPointF(x, y))

            painter.drawPolyline(poly)

        draw_curve(self.hist_r, (255, 80, 80), max_r)
        draw_curve(self.hist_g, (80, 255, 80), max_g)
        draw_curve(self.hist_b, (80, 80, 255), max_b)

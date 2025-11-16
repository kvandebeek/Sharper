from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF, QMouseEvent
from PySide6.QtCore import Qt, QPointF, QPoint
import numpy as np


class HistogramWidget(QWidget):
    def __init__(self, parent=None, threshold=0.05):
        super().__init__(parent)

        self.hist_r = None
        self.hist_g = None
        self.hist_b = None
        self.threshold = threshold

        # visibility toggle
        self.visible = True

        # dragging
        self.dragging = False
        self.drag_offset = QPoint(0, 0)

        # ensure hist receives mouse events
        self.setMouseTracking(True)

    # ----------------------------------------
    # HIDE / SHOW
    # ----------------------------------------
    def toggle_visibility(self):
        self.visible = not self.visible
        self.setVisible(self.visible)
        self.update()

    # ----------------------------------------
    # DRAGGING LOGIC
    # ----------------------------------------
    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_offset = ev.pos()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            self.dragging = False

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.dragging:
            parent = self.parent()
            if parent:
                new_pos = self.mapToParent(ev.pos() - self.drag_offset)

                # keep inside bounds of parent
                px = max(0, min(new_pos.x(), parent.width() - self.width()))
                py = max(0, min(new_pos.y(), parent.height() - self.height()))

                self.move(px, py)

    # ----------------------------------------
    # HISTOGRAM UPDATE
    # ----------------------------------------
    def update_image(self, img):
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            return

        # mask out background (signal-only histogram)
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

        self.hist_r, _ = np.histogram(r, 256, (0, 255))
        self.hist_g, _ = np.histogram(g, 256, (0, 255))
        self.hist_b, _ = np.histogram(b, 256, (0, 255))

        self.update()

    # ----------------------------------------
    # DRAW HISTOGRAM
    # ----------------------------------------
    def paintEvent(self, event):
        if not self.visible:
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        if self.hist_r is None:
            return

        w = self.width()
        h = self.height()

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

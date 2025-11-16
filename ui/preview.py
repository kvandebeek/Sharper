from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLabel, QSlider
)
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import Qt, Signal, QEvent

import numpy as np
from ui.histogram import HistogramWidget


class PreviewWidget(QWidget):
    pixel_info = Signal(int, int, float, float, float)

    def __init__(self):
        super().__init__()

        self.scale_factor = 1.0
        self.min_scale = 1 / 8
        self.max_scale = 4.0

        self.img_array = None

        layout = QVBoxLayout(self)

        # ==============================================================
        # GRAPHICS VIEW
        # ==============================================================
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)

        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setMouseTracking(True)

        self.view.setRenderHints(
            self.view.renderHints()
            | QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
        )

        layout.addWidget(self.view)

        # ==============================================================
        # HISTOGRAM OVERLAY
        # ==============================================================
        self.hist_overlay = HistogramWidget(self.view.viewport())
        self.hist_overlay.resize(240, 150)
        self.hist_overlay.move(10, 10)
        self.hist_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.hist_overlay.raise_()

        # ==============================================================
        # ZOOM SLIDER
        # ==============================================================
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 32)
        self.zoom_slider.setValue(8)  # 1x zoom
        self.zoom_slider.valueChanged.connect(self._zoom_from_slider)

        zoom_layout.addWidget(self.zoom_slider)
        layout.addLayout(zoom_layout)

        # ==============================================================
        # EVENT FILTERS + KEY FOCUS
        # ==============================================================
        self.setFocusPolicy(Qt.StrongFocus)
        self.view.setFocusPolicy(Qt.ClickFocus)

        self.view.viewport().installEventFilter(self)

    # ==============================================================
    # IMAGE UPDATE
    # ==============================================================
    def update_image(self, img_uint8):
        """
        img_uint8: uint8 RGB
        """
        self.img_array = img_uint8.astype(np.float32) / 255.0

        h, w, _ = img_uint8.shape
        qimg = QImage(img_uint8.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item.setPixmap(pixmap)

        self._apply_zoom()

    # ==============================================================
    # HISTOGRAM UPDATE
    # ==============================================================
    def update_histogram(self):
        if self.img_array is not None:
            self.hist_overlay.update_image(self.img_array)
            self.hist_overlay.raise_()

    # ==============================================================
    # EVENT FILTER (mouse move + wheel zoom)
    # ==============================================================
    def eventFilter(self, obj, ev):
        if self.img_array is None:
            return super().eventFilter(obj, ev)

        et = ev.type()

        # ----------------------------------------------------------
        # Pixel readout
        # ----------------------------------------------------------
        if et == QEvent.MouseMove:
            pos = ev.position()
            scene_pos = self.view.mapToScene(int(pos.x()), int(pos.y()))
            x = int(scene_pos.x())
            y = int(scene_pos.y())

            if (
                0 <= y < self.img_array.shape[0]
                and 0 <= x < self.img_array.shape[1]
            ):
                r, g, b = self.img_array[y, x]
                self.pixel_info.emit(x, y, r, g, b)

            return True

        # ----------------------------------------------------------
        # Mouse wheel zoom
        # ----------------------------------------------------------
        if et == QEvent.Wheel:
            delta = ev.angleDelta().y()
            if delta > 0:
                self._zoom(1.1)
            else:
                self._zoom(0.9)
            return True

        return super().eventFilter(obj, ev)

    # ==============================================================
    # KEYBOARD SHORTCUTS
    # ==============================================================    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_H:
            self.hist_overlay.toggle_visibility()
            return
        super().keyPressEvent(event)

    # ==============================================================
    # ZOOM SYSTEM
    # ==============================================================
    def _zoom(self, factor):
        new_scale = self.scale_factor * factor
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        self.scale_factor = new_scale

        slider_val = int(self.scale_factor * 8)
        slider_val = max(1, min(32, slider_val))

        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(slider_val)
        self.zoom_slider.blockSignals(False)

        self._apply_zoom()

    def _zoom_from_slider(self, v):
        self.scale_factor = v / 8.0
        self.scale_factor = max(self.min_scale, min(self.max_scale, self.scale_factor))
        self._apply_zoom()

    def _apply_zoom(self):
        if self.pixmap_item.pixmap().isNull():
            return

        self.view.resetTransform()
        self.view.scale(self.scale_factor, self.scale_factor)
        self.hist_overlay.raise_()

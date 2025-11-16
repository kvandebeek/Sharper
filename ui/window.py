import sys
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QLabel, QFileDialog, QMessageBox, QToolBar,
    QPushButton, QProgressBar
)
from PySide6.QtCore import Qt

from ui.preview import PreviewWidget
from ui.controls import ControlsPanel


class MainWindow(QMainWindow):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.base_rgb = None
        self.live_updates_enabled = False

        self._build_ui()
        self.setWindowTitle("Sharper")

    # ------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)

        # PREVIEW
        self.preview = PreviewWidget()
        self.preview.pixel_info.connect(self._update_pixel_info)
        layout.addWidget(self.preview, stretch=3)

        # CONTROLS
        self.controls = ControlsPanel(self.pipeline)
        self.controls.params_changed.connect(self._rerun_partial)
        self.controls.reset_requested.connect(self._reset_all)
        layout.addWidget(self.controls, stretch=1)

        self.setCentralWidget(central)

        # STATUS BAR
        self.status_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.statusBar().addPermanentWidget(self.status_label, 2)
        self.statusBar().addPermanentWidget(self.progress_bar, 1)

        # TOOLBAR
        tb = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, tb)

        open_btn = QPushButton("Open")
        save_btn = QPushButton("Save")
        open_btn.clicked.connect(self._open_image)
        save_btn.clicked.connect(self._save_result)
        tb.addWidget(open_btn)
        tb.addWidget(save_btn)

    # ------------------------------------------------------------
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not path:
            return

        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            QMessageBox.warning(self, "Error", "Could not load image.")
            return

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        self.base_rgb = arr.copy()
        self.preview.update_image(arr)
        self.preview.update_histogram()

        self.progress_bar.setValue(0)
        self.live_updates_enabled = True
        self.status_label.setText(f"Loaded: {path}")

    # ------------------------------------------------------------
    def _update_pixel_info(self, x, y, r, g, b):
        self.status_label.setText(
            f"X:{x}   Y:{y}   R:{r:.3f}  G:{g:.3f}  B:{b:.3f}"
        )

    # ------------------------------------------------------------
    def _rerun_partial(self):
        if not self.live_updates_enabled:
            return
        self.status_label.setText("Processing…")
        self._apply_update()

    def _update_progress(self, pct, txt):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(txt)

    def _apply_update(self):
        if self.base_rgb is None:
            return

        proc = self.pipeline.apply_all(self.base_rgb, self._update_progress)
        out = (proc * 255).clip(0, 255).astype(np.uint8)

        self.preview.update_image(out)
        self.preview.update_histogram()
        self.status_label.setText("Live update")

    # ------------------------------------------------------------
    def _reset_all(self):
        if self.base_rgb is None:
            return

        self.controls.reset_to_defaults()
        self.preview.update_image(self.base_rgb)
        self.preview.update_histogram()
        self.status_label.setText("Reset")
        self.progress_bar.setValue(0)

    # ------------------------------------------------------------
    def _save_result(self):
        if self.base_rgb is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "", "PNG Files (*.png)"
        )
        if not path:
            return

        proc = self.pipeline.apply_all(self.base_rgb, self._update_progress)
        out = (proc * 255).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out_bgr)

        self.progress_bar.setValue(100)
        self.status_label.setText(f"Saved: {path}")


def start_qt(pipeline):
    app = QApplication(sys.argv)
    win = MainWindow(pipeline)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())

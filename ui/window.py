import sys
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QLabel,
    QFileDialog, QMessageBox, QApplication, QToolBar, QPushButton
)
from PySide6.QtCore import Qt

from ui.preview import PreviewWidget
from ui.controls import ControlsPanel


class MainWindow(QMainWindow):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.base_rgb = None  # uint8, H×W×3
        self.live_updates_enabled = False

        self._build_ui()
        self.setWindowTitle("Planetary Enhancer")

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)

        self.preview = PreviewWidget()
        layout.addWidget(self.preview, stretch=3)

        self.controls = ControlsPanel(self.pipeline)
        self.controls.params_changed.connect(self._rerun_partial)
        self.controls.reset_requested.connect(self._reset_all)

        layout.addWidget(self.controls, stretch=1)
        self.setCentralWidget(central)

        self.status_label = QLabel("")
        self.statusBar().addPermanentWidget(self.status_label)

        tb = QToolBar("Main", self)
        self.addToolBar(Qt.TopToolBarArea, tb)

        open_btn = QPushButton("Open")
        save_btn = QPushButton("Save")
        open_btn.clicked.connect(self._open_image)
        save_btn.clicked.connect(self._save_result)
        tb.addWidget(open_btn)
        tb.addWidget(save_btn)

    # ------------------ image loading ------------------
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not path:
            return

        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            QMessageBox.warning(self, "Error", "Could not load image.")
            return

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]

        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            m = float(arr.max()) or 1.0
            arr = (arr / m * 255.0).clip(0, 255).astype(np.uint8)
            arr = np.stack([arr, arr, arr], axis=2)

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.base_rgb = arr.astype(np.uint8)

        self.preview.update_image(self.base_rgb)
        self.live_updates_enabled = True
        self.status_label.setText(f"Loaded: {path}")

    # ------------------ live preview ------------------
    def _rerun_partial(self):
        if not self.live_updates_enabled or self.base_rgb is None:
            return
        proc = self.pipeline.apply_all(self.base_rgb)
        out = (proc * 255.0).clip(0, 255).astype(np.uint8)
        self.preview.update_image(out)
        self.status_label.setText("Live update")

    # ------------------ reset all controls ------------------
    def _reset_all(self):
        if not self.live_updates_enabled:
            return
        self.controls.reset_to_defaults()
        if self.base_rgb is not None:
            self.preview.update_image(self.base_rgb)
        self.status_label.setText("Reset")

    # ------------------ save result ------------------
    def _save_result(self):
        if self.base_rgb is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "", "PNG Files (*.png)"
        )
        if not path:
            return

        proc = self.pipeline.apply_all(self.base_rgb)
        out = (proc * 255.0).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out_bgr)
        self.status_label.setText(f"Saved: {path}")


def start_qt(pipeline):
    app = QApplication(sys.argv)
    win = MainWindow(pipeline)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())

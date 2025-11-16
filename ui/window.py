import sys
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QLabel,
    QFileDialog, QMessageBox, QApplication, QToolBar, QPushButton
)
from PySide6.QtCore import Qt, QTimer

from ui.preview import PreviewWidget
from ui.controls import ControlsPanel


class MainWindow(QMainWindow):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.base_rgb = None
        self.live_updates_enabled = False

        # Debounce timer for live updates
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._apply_update)

        self._build_ui()
        self.setWindowTitle("Sharper")

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # IMAGE LOADING
    # ------------------------------------------------------------
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not path:
            return

        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            QMessageBox.warning(self, "Error", "Could not load image.")
            return

        print("RAW LOAD:", arr.shape, arr.dtype, arr.min(), arr.max())

        # 1) Handle 16-bit inputs
        if arr.dtype == np.uint16:
            maxv = float(arr.max()) if arr.max() > 0 else 1.0
            arr = arr.astype(np.float32) / maxv  # now 0..1

        # 2) Handle fake RGB mono TIFFs
        if arr.ndim == 3 and arr.shape[2] == 3:
            if np.allclose(arr[..., 0], arr[..., 1], atol=1e-6) and \
               np.allclose(arr[..., 0], arr[..., 2], atol=1e-6):
                gray = arr[..., 0]
                arr = np.stack([gray, gray, gray], axis=2)

        # 3) Convert real grayscale to RGB
        if arr.ndim == 2:
            m = float(arr.max()) if arr.max() > 0 else 1.0
            arr = arr.astype(np.float32) / m
            arr = np.stack([arr, arr, arr], axis=2)

        # 4) Normalize everything to uint8 RGB
        if arr.max() <= 1.5:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        # strip alpha channel if any
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        self.base_rgb = arr.copy()
        self.preview.update_image(arr)
        self.live_updates_enabled = True
        self.status_label.setText(f"Loaded: {path}")

    # ------------------------------------------------------------
    # LIVE PREVIEW (DEBOUNCED)
    # ------------------------------------------------------------
    def _rerun_partial(self):
        """Called whenever a slider changes; start/restart debounce timer."""
        if not self.live_updates_enabled:
            return
        # restart 225 ms timer
        self.update_timer.start(225)

    def _apply_update(self):
        """Actually run the pipeline and update preview."""
        if not self.live_updates_enabled or self.base_rgb is None:
            return
        proc = self.pipeline.apply_all(self.base_rgb)
        out = (proc * 255.0).clip(0, 255).astype(np.uint8)
        self.preview.update_image(out)
        self.status_label.setText("Live update")

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------
    def _reset_all(self):
        if not self.live_updates_enabled:
            return
        self.controls.reset_to_defaults()
        if self.base_rgb is not None:
            self.preview.update_image(self.base_rgb)
        self.status_label.setText("Reset")

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------
    def _save_result(self):
        if self.base_rgb is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output",
            "",
            "PNG Files (*.png)"
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

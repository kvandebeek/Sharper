from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QSlider, QPushButton, QHBoxLayout
)
from PySide6.QtCore import Qt, Signal


class ControlsPanel(QWidget):
    params_changed = Signal()
    reset_requested = Signal()

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_preblur_tab()
        self._build_decon_tab()
        self._build_wavelets_tab()
        self._build_detail_tab()
        self._build_denoise_tab()
        self._build_final_tab()

        reset_layout = QHBoxLayout()
        reset_layout.setAlignment(Qt.AlignRight)
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self._on_reset_clicked)
        reset_layout.addWidget(reset_btn)
        layout.addLayout(reset_layout)

        layout.addStretch()

    # ---------- helpers ----------
    def _make_slider(self, minimum, maximum, value):
        s = QSlider(Qt.Horizontal)
        s.setRange(minimum, maximum)
        s.setValue(value)
        return s

    def _on_reset_clicked(self):
        self.reset_to_defaults()
        self.params_changed.emit()
        self.reset_requested.emit()

    # ---------- public for window ----------
    def reset_to_defaults(self):
        # pipeline: 0 PreBlur, 1 Preprocess, 2 Decon, 3 Wavelets, 4 Detail, 5 NR, 6 Final
        pb = self.pipeline.items[0]
        dc = self.pipeline.items[2]
        wa = self.pipeline.items[3]
        de = self.pipeline.items[4]
        nr = self.pipeline.items[5]
        fa = self.pipeline.items[6]

        pb.sigma = 0.0
        dc.iterations = 0
        dc.psf_sigma = 1.5
        wa.gains = [0.0, 0.0, 0.0, 0.0, 0.0]
        de.amount = 0.0
        de.radius = 2.0
        nr.strength = 0.0
        nr.radius = 1.0
        fa.gamma = 1.0
        fa.contrast = 1.0
        fa.brightness = 0.0

        # sync sliders
        self.preblur_slider.setValue(0)
        self.decon_iter_slider.setValue(0)
        self.decon_sigma_slider.setValue(15)
        for s in self.wavelet_sliders:
            s.setValue(0)
        self.detail_amount_slider.setValue(0)
        self.detail_radius_slider.setValue(20)
        self.denoise_strength_slider.setValue(0)
        self.denoise_radius_slider.setValue(10)
        self.final_gamma_slider.setValue(100)
        self.final_contrast_slider.setValue(100)
        self.final_brightness_slider.setValue(0)

        # update labels
        self._update_preblur_label()
        self._update_decon_labels()
        self._update_wavelet_labels()
        self._update_detail_labels()
        self._update_denoise_labels()
        self._update_final_labels()

    # ---------- PreBlur tab ----------
    def _build_preblur_tab(self):
        from workflows.preblur import PreBlur  # just for type hints, not strictly needed
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.preblur_label = QLabel()
        self.preblur_slider = self._make_slider(0, 100, int(self.pipeline.items[0].sigma * 20.0))
        self.preblur_slider.valueChanged.connect(self._preblur_changed)

        layout.addWidget(self.preblur_label)
        layout.addWidget(self.preblur_slider)
        layout.addStretch()

        self.tabs.addTab(tab, "Pre-Blur")
        self._update_preblur_label()

    def _update_preblur_label(self):
        sigma = self.preblur_slider.value() / 20.0
        self.preblur_label.setText(f"Sigma: {sigma:.2f}")

    def _preblur_changed(self, v):
        sigma = v / 20.0
        self.pipeline.items[0].sigma = sigma
        self._update_preblur_label()
        self.params_changed.emit()

    # ---------- Decon tab ----------
    def _build_decon_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.decon_iter_label = QLabel()
        self.decon_iter_slider = self._make_slider(0, 50, self.pipeline.items[2].iterations)

        self.decon_sigma_label = QLabel()
        self.decon_sigma_slider = self._make_slider(5, 50, int(self.pipeline.items[2].psf_sigma * 10.0))

        self.decon_iter_slider.valueChanged.connect(self._decon_changed)
        self.decon_sigma_slider.valueChanged.connect(self._decon_changed)

        layout.addWidget(self.decon_iter_label)
        layout.addWidget(self.decon_iter_slider)
        layout.addWidget(self.decon_sigma_label)
        layout.addWidget(self.decon_sigma_slider)
        layout.addStretch()

        self.tabs.addTab(tab, "Deconvolution")
        self._update_decon_labels()

    def _update_decon_labels(self):
        it = self.decon_iter_slider.value()
        sigma = self.decon_sigma_slider.value() / 10.0
        self.decon_iter_label.setText(f"Iterations: {it}")
        self.decon_sigma_label.setText(f"PSF Sigma: {sigma:.2f}")

    def _decon_changed(self, _):
        it = self.decon_iter_slider.value()
        sigma = self.decon_sigma_slider.value() / 10.0
        dc = self.pipeline.items[2]
        dc.iterations = it
        dc.psf_sigma = sigma
        self._update_decon_labels()
        self.params_changed.emit()

    # ---------- Wavelets tab ----------
    def _build_wavelets_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.wavelet_sliders = []
        self.wavelet_labels = []

        wa = self.pipeline.items[3]
        for i in range(5):
            label = QLabel()
            slider = self._make_slider(0, 200, int(wa.gains[i] * 100.0))
            slider.valueChanged.connect(self._wavelets_changed)
            layout.addWidget(label)
            layout.addWidget(slider)
            self.wavelet_labels.append(label)
            self.wavelet_sliders.append(slider)

        layout.addStretch()
        self.tabs.addTab(tab, "Wavelets")
        self._update_wavelet_labels()

    def _update_wavelet_labels(self):
        for i, (label, slider) in enumerate(zip(self.wavelet_labels, self.wavelet_sliders)):
            gain = slider.value() / 100.0
            label.setText(f"Level {i+1}: {gain:.2f}")

    def _wavelets_changed(self, _):
        wa = self.pipeline.items[3]
        wa.gains = [s.value() / 100.0 for s in self.wavelet_sliders]
        self._update_wavelet_labels()
        self.params_changed.emit()


    # ---------- Detail tab ----------
    def _build_detail_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.detail_amount_label = QLabel()
        self.detail_amount_slider = self._make_slider(0, 300, int(self.pipeline.items[4].amount * 100.0))

        self.detail_radius_label = QLabel()
        self.detail_radius_slider = self._make_slider(5, 100, int(self.pipeline.items[4].radius * 10.0))

        self.detail_amount_slider.valueChanged.connect(self._detail_changed)
        self.detail_radius_slider.valueChanged.connect(self._detail_changed)

        layout.addWidget(self.detail_amount_label)
        layout.addWidget(self.detail_amount_slider)
        layout.addWidget(self.detail_radius_label)
        layout.addWidget(self.detail_radius_slider)
        layout.addStretch()

        self.tabs.addTab(tab, "Detail")
        self._update_detail_labels()

    def _update_detail_labels(self):
        amt = self.detail_amount_slider.value() / 100.0
        rad = self.detail_radius_slider.value() / 10.0
        self.detail_amount_label.setText(f"Amount: {amt:.2f}")
        self.detail_radius_label.setText(f"Radius: {rad:.2f}")

    def _detail_changed(self, _):
        amt = self.detail_amount_slider.value() / 100.0
        rad = self.detail_radius_slider.value() / 10.0
        de = self.pipeline.items[4]
        de.amount = amt
        de.radius = rad
        self._update_detail_labels()
        self.params_changed.emit()

    # ---------- Denoise tab ----------
    def _build_denoise_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.denoise_strength_label = QLabel()
        self.denoise_strength_slider = self._make_slider(0, 100, int(self.pipeline.items[5].strength))

        self.denoise_radius_label = QLabel()
        self.denoise_radius_slider = self._make_slider(10, 200, int(self.pipeline.items[5].radius * 10.0))

        self.denoise_strength_slider.valueChanged.connect(self._denoise_changed)
        self.denoise_radius_slider.valueChanged.connect(self._denoise_changed)

        layout.addWidget(self.denoise_strength_label)
        layout.addWidget(self.denoise_strength_slider)
        layout.addWidget(self.denoise_radius_label)
        layout.addWidget(self.denoise_radius_slider)
        layout.addStretch()

        self.tabs.addTab(tab, "Denoise")
        self._update_denoise_labels()

    def _update_denoise_labels(self):
        st = self.denoise_strength_slider.value()
        rad = self.denoise_radius_slider.value() / 10.0
        self.denoise_strength_label.setText(f"Strength: {st:.1f}")
        self.denoise_radius_label.setText(f"Radius: {rad:.2f}")

    def _denoise_changed(self, _):
        nr = self.pipeline.items[5]
        nr.strength = float(self.denoise_strength_slider.value())
        nr.radius = self.denoise_radius_slider.value() / 10.0
        self._update_denoise_labels()
        self.params_changed.emit()

    # ---------- Final tab ----------
    def _build_final_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.final_gamma_label = QLabel()
        self.final_gamma_slider = self._make_slider(30, 300, int(self.pipeline.items[6].gamma * 100.0))

        self.final_contrast_label = QLabel()
        self.final_contrast_slider = self._make_slider(30, 300, int(self.pipeline.items[6].contrast * 100.0))

        self.final_brightness_label = QLabel()
        self.final_brightness_slider = self._make_slider(-50, 50, int(self.pipeline.items[6].brightness * 100.0))

        self.final_gamma_slider.valueChanged.connect(self._final_changed)
        self.final_contrast_slider.valueChanged.connect(self._final_changed)
        self.final_brightness_slider.valueChanged.connect(self._final_changed)

        layout.addWidget(self.final_gamma_label)
        layout.addWidget(self.final_gamma_slider)
        layout.addWidget(self.final_contrast_label)
        layout.addWidget(self.final_contrast_slider)
        layout.addWidget(self.final_brightness_label)
        layout.addWidget(self.final_brightness_slider)
        layout.addStretch()

        self.tabs.addTab(tab, "Final")
        self._update_final_labels()

    def _update_final_labels(self):
        ga = self.final_gamma_slider.value() / 100.0
        co = self.final_contrast_slider.value() / 100.0
        br = self.final_brightness_slider.value() / 100.0
        self.final_gamma_label.setText(f"Gamma: {ga:.2f}")
        self.final_contrast_label.setText(f"Contrast: {co:.2f}")
        self.final_brightness_label.setText(f"Brightness: {br:.2f}")

    def _final_changed(self, _):
        ga = self.final_gamma_slider.value() / 100.0
        co = self.final_contrast_slider.value() / 100.0
        br = self.final_brightness_slider.value() / 100.0
        fa = self.pipeline.items[6]
        fa.gamma = ga
        fa.contrast = co
        fa.brightness = br
        self._update_final_labels()
        self.params_changed.emit()

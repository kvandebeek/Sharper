import numpy as np
from core.workflow_item import WorkflowItem


class Deconvolution(WorkflowItem):
    """
    FFT-based Richardson–Lucy deconvolution, per channel.

    - Works on 2D (mono) or 3D (RGB) float images in [0,1].
    - Optional progress_cb(iter_idx, total_iters) for UI progress.
    """

    def __init__(self, image_ref, iterations=0, psf_sigma=1.5):
        super().__init__("Deconvolution")
        self.image_ref = image_ref
        self.iterations = iterations  # 0 = off
        self.psf_sigma = psf_sigma

    # -----------------------------------------------------------
    # Gaussian PSF
    # -----------------------------------------------------------
    def _make_gaussian_psf(self, sigma):
        size = int(max(7, sigma * 6.0))
        if size % 2 == 0:
            size += 1

        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        psf /= psf.sum().clip(1e-12)

        return psf.astype(np.float32)

    def _prepare_psf_fft(self, psf, shape):
        """
        Pad and center PSF for FFT-based convolution.
        """
        h, w = shape
        ph, pw = psf.shape

        psf_padded = np.zeros((h, w), dtype=np.float32)
        psf_padded[:ph, :pw] = psf

        psf_padded = np.roll(psf_padded, -ph // 2, axis=0)
        psf_padded = np.roll(psf_padded, -pw // 2, axis=1)

        H = np.fft.rfftn(psf_padded)
        H_conj = np.conj(H)
        return H, H_conj

    # -----------------------------------------------------------
    # FFT-based RL for a single channel
    # -----------------------------------------------------------
    def _rl_fft_channel(self, ch, psf, iters, progress_cb=None):
        ch = ch.astype(np.float32)
        ch = np.clip(ch, 0.0, 1.0)

        if iters <= 0:
            return ch

        shape = ch.shape
        eps = 1e-8

        H, H_conj = self._prepare_psf_fft(psf, shape)

        estimate = ch.copy()

        for i in range(iters):
            # conv(estimate, psf)
            E = np.fft.rfftn(estimate)
            conv_est = np.fft.irfftn(E * H, s=shape)

            conv_est = np.clip(conv_est, eps, None)
            relative_blur = ch / conv_est

            # conv(relative_blur, psf_mirror)
            RB = np.fft.rfftn(relative_blur)
            corr = np.fft.irfftn(RB * H_conj, s=shape)

            estimate *= corr

            estimate = np.nan_to_num(estimate, nan=0.0, posinf=1.0, neginf=0.0)
            estimate = np.clip(estimate, 0.0, 2.0)

            if progress_cb is not None:
                progress_cb(i + 1, iters)

        estimate = np.clip(estimate, 0.0, 1.0)
        return estimate

    # -----------------------------------------------------------
    # APPLY
    # -----------------------------------------------------------
    def apply(self, img, progress_cb=None):
        img = img.astype(np.float32)

        if self.iterations <= 0 or self.psf_sigma <= 0.0:
            return img

        psf = self._make_gaussian_psf(self.psf_sigma)

        if img.ndim == 2:
            return self._rl_fft_channel(img, psf, self.iterations, progress_cb)

        h, w, c = img.shape
        out = np.empty_like(img)
        for ch_idx in range(c):
            out[..., ch_idx] = self._rl_fft_channel(
                img[..., ch_idx], psf, self.iterations, progress_cb
            )
        return out

    # -----------------------------------------------------------
    # EXECUTE
    # -----------------------------------------------------------
    def execute(self):
        arr = self.image_ref[0]
        if arr is None:
            return
        self.image_ref[0] = self.apply(arr)

import numpy as np
import cv2
from core.workflow_item import WorkflowItem


class Wavelets(WorkflowItem):
    """
    Multiscale Gaussian (DoG) wavelets, per-channel, fast implementation.

    - Uses separable Gaussian (getGaussianKernel + sepFilter2D) for speed.
    - Base is the original image, so positive gains sharpen instead of blur.
    - Works on 2D (mono) or 3D (RGB) float images in [0,1].
    """

    def __init__(self, image_ref, gains=None):
        super().__init__("Wavelets")
        self.image_ref = image_ref
        # 5 levels, all 0 = disabled
        self.gains = gains or [0.0, 0.0, 0.0, 0.0, 0.0]
        # Reasonable sigmas for planetary detail (pixels)
        self.sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]

    def _gaussian_blur_fast(self, ch, sigma):
        """Fast separable Gaussian blur on a single channel."""
        if sigma <= 0.0:
            return ch

        # Kernel size: ~6 sigma, odd, at least 3
        ksize = int(max(3, sigma * 6.0))
        if ksize % 2 == 0:
            ksize += 1

        k = cv2.getGaussianKernel(ksize, sigma)
        blurred = cv2.sepFilter2D(
            ch, -1, k, k, borderType=cv2.BORDER_REFLECT_101
        )
        return blurred

    def _process_channel(self, ch):
        ch = ch.astype(np.float32)

        # If all gains are zero, no-op
        if not any(self.gains):
            return ch

        # Gaussian pyramid
        gauss = [ch]
        for sigma in self.sigmas:
            g = self._gaussian_blur_fast(ch, sigma)
            gauss.append(g)

        # Detail layers as differences between scales
        details = [gauss[i] - gauss[i + 1] for i in range(len(self.sigmas))]

        # Base = original channel (sharpening)
        out = ch.copy()

        # Apply weighted details
        for i, gain in enumerate(self.gains):
            if gain == 0.0:
                continue
            out += details[i] * float(gain)

        # Sanitize
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out, 0.0, 1.0)

    def apply(self, img):
        img = img.astype(np.float32)

        if img.ndim == 2:
            return self._process_channel(img)

        # RGB
        h, w, c = img.shape
        out = np.empty_like(img)
        for ch_idx in range(c):
            out[..., ch_idx] = self._process_channel(img[..., ch_idx])

        return out

    def execute(self):
        self.image_ref[0] = self.apply(self.image_ref[0])

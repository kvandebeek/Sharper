import numpy as np
import cv2
from core.workflow_item import WorkflowItem

class Deconvolution(WorkflowItem):
    def __init__(self, image_ref, iterations=0, psf_sigma=1.5):
        super().__init__("Deconvolution")
        self.image_ref = image_ref
        self.iterations = iterations  # 0 = off
        self.psf_sigma = psf_sigma

    def _make_gaussian_psf(self, sigma):
        size = int(max(7, sigma * 6.0))
        if size % 2 == 0: size += 1
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx*xx + yy*yy) / (2.0 * sigma * sigma))
        psf /= psf.sum()
        return psf.astype(np.float32)

    def _rl_channel(self, ch, psf, iters):
        ch = ch.astype(np.float32).clip(0.0, 1.0)
        estimate = ch.copy()
        psf_m = psf[::-1, ::-1]
        eps = 1e-8
        for _ in range(iters):
            conv_est = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT_101)
            conv_est = np.clip(conv_est, eps, None)
            rel = ch / conv_est
            corr = cv2.filter2D(rel, -1, psf_m, borderType=cv2.BORDER_REFLECT_101)
            estimate *= corr
            estimate = np.nan_to_num(estimate, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 2.0)
        return estimate.clip(0.0, 1.0)

    def apply(self, img):
        if self.iterations <= 0 or self.psf_sigma <= 0.0:
            return img.astype(np.float32)
        img = img.astype(np.float32)
        psf = self._make_gaussian_psf(self.psf_sigma)
        if img.ndim == 2:
            return self._rl_channel(img, psf, self.iterations)
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = self._rl_channel(img[..., c], psf, self.iterations)
        return out

    def execute(self):
        self.image_ref[0] = self.apply(self.image_ref[0])

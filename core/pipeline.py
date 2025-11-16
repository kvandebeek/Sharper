import numpy as np

from workflows.preblur import PreBlur
from workflows.preprocess import Preprocess
from workflows.deconvolution import Deconvolution
from workflows.wavelets import Wavelets
from workflows.detail_boost import DetailBoost
from workflows.noise_reduction import NoiseReduction
from workflows.final_adjust import FinalAdjust


class Pipeline:
    def __init__(self):
        dummy = [None]
        self.items = [
            PreBlur(dummy),         # 0
            Preprocess(dummy),      # 1
            Deconvolution(dummy),   # 2
            Wavelets(dummy),        # 3
            DetailBoost(dummy),     # 4
            NoiseReduction(dummy),  # 5
            FinalAdjust(dummy)      # 6
        ]

    def _run_chain(self, img):
        out = img.astype(np.float32)
        for item in self.items:
            out = item.apply(out)
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = out.clip(0.0, 1.0)
        return out

    def apply_all(self, img_rgb):
        if img_rgb is None:
            return None

        arr = img_rgb.astype(np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        arr = arr.clip(0.0, 1.0)

        if arr.ndim == 3 and arr.shape[2] == 3:
            R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
            if np.allclose(R, G, atol=1e-5) and np.allclose(R, B, atol=1e-5):
                gray = R
                out = self._run_chain(gray)
                return np.stack([out, out, out], axis=2)

            return self._run_chain(arr)

        if arr.ndim == 2:
            out = self._run_chain(arr)
            return np.stack([out, out, out], axis=2)

        out = self._run_chain(arr)
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=2)
        return out

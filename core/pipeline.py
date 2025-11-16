from workflows.preblur import PreBlur
from workflows.preprocess import Preprocess
from workflows.deconvolution import Deconvolution
from workflows.wavelets import Wavelets
from workflows.detail_boost import DetailBoost
from workflows.noise_reduction import NoiseReduction
from workflows.final_adjust import FinalAdjust

import numpy as np


class Pipeline:
    def __init__(self):
        dummy_ref = [None]  # kept to satisfy constructors, but not used
        self.items = [
            PreBlur(dummy_ref),         # 0
            Preprocess(dummy_ref),      # 1
            Deconvolution(dummy_ref),   # 2
            Wavelets(dummy_ref),        # 3
            DetailBoost(dummy_ref),     # 4
            NoiseReduction(dummy_ref),  # 5
            FinalAdjust(dummy_ref)      # 6
        ]

    def _run_chain(self, img):
        """Run all workflow items on img (2D or 3D float32, 0..1)."""
        out = img
        for item in self.items:
            out = item.apply(out)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out, 0.0, 1.0)

    def apply_all(self, img_rgb):
        """
        img_rgb: uint8 H×W×3 (from window.py)
        Returns float32 H×W×3, 0..1 for preview/saving.
        Handles:
          - true color images
          - grayscale images (kept grayscale)
        """
        if img_rgb is None:
            return None

        arr = img_rgb.astype(np.float32)

        # Normalize if it looks like 0..255
        maxv = float(arr.max()) if arr.size else 1.0
        if maxv > 1.5:
            arr /= 255.0

        # If already 0..1, keep as-is
        arr = np.clip(arr, 0.0, 1.0)

        # Detect grayscale: all three channels nearly equal
        if arr.ndim == 3 and arr.shape[2] == 3:
            R = arr[..., 0]
            G = arr[..., 1]
            B = arr[..., 2]

            if np.allclose(R, G, atol=1e-5) and np.allclose(R, B, atol=1e-5):
                # Treat as single-channel image
                gray = R.copy()
                gray = np.clip(gray, 0.0, 1.0)
                gray_out = self._run_chain(gray)  # 2D
                # Replicate to 3 channels for preview
                rgb_out = np.stack([gray_out, gray_out, gray_out], axis=2)
                return rgb_out

            # True color image: run chain on full RGB
            rgb_out = self._run_chain(arr)
            return rgb_out

        # Safety fallback: unexpected shape; run chain as-is and try to make 3-ch
        out = self._run_chain(arr)
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=2)
        return out

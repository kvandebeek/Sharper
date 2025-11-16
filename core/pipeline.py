import numpy as np

from workflows.preblur import PreBlur
from workflows.preprocess import Preprocess
from workflows.deconvolution import Deconvolution
from workflows.wavelets import Wavelets
from workflows.detail_boost import DetailBoost
from workflows.noise_reduction import NoiseReduction
from workflows.final_adjust import FinalAdjust


class Pipeline:
    """
    Runs the full RGB/mono pipeline and optionally reports progress.

    progress_cb(percent, text):
        - percent: float 0..100
        - text: short status string
    """

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

        # Weighted contribution of each step to total progress
        # indexes: [PreBlur, Preprocess, Decon, Wavelets, Detail, Denoise, Final]
        self.weights = [5.0, 5.0, 50.0, 25.0, 7.0, 5.0, 3.0]  # sum = 100

    # -------------------------------------------------------
    # Internal: run chain on mono or RGB
    # -------------------------------------------------------
    def _run_chain(self, img, progress_cb=None):
        out = img.astype(np.float32)
        progress = 0.0

        for idx, item in enumerate(self.items):
            w = self.weights[idx]
            name = getattr(item, "name", f"Step {idx}")

            # Special handling for Deconvolution: forward per-iteration progress
            if isinstance(item, Deconvolution) and getattr(item, "iterations", 0) > 0:
                if progress_cb is not None:
                    base_progress = progress
                    total_w = w

                    def iter_cb(iter_idx, total_iters, base=base_progress, step_w=total_w):
                        frac = iter_idx / float(max(total_iters, 1))
                        percent = base + step_w * frac
                        progress_cb(percent, f"Deconvolution {iter_idx}/{total_iters}")

                else:
                    iter_cb = None

                out = item.apply(out, iter_cb)
                progress += w
                if progress_cb is not None:
                    progress_cb(progress, "Deconvolution done")
            else:
                out = item.apply(out)
                progress += w
                if progress_cb is not None:
                    progress_cb(progress, name)

            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0)

        # weights sum to 100, so progress ends at ~100
        return out

    # -------------------------------------------------------
    # Public: run pipeline on an RGB image
    # -------------------------------------------------------
    def apply_all(self, img_rgb, progress_cb=None):
        """
        img_rgb: uint8 H×W×3
        Returns float32 H×W×3 in [0,1].
        """
        if img_rgb is None:
            return None

        arr = img_rgb.astype(np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        arr = arr.clip(0.0, 1.0)

        # RGB image?
        if arr.ndim == 3 and arr.shape[2] == 3:
            R = arr[..., 0]
            G = arr[..., 1]
            B = arr[..., 2]

            # Grayscale disguised as RGB?
            if np.allclose(R, G, atol=1e-5) and np.allclose(R, B, atol=1e-5):
                gray = R
                out = self._run_chain(gray, progress_cb)
                return np.stack([out, out, out], axis=2)

            # True color
            return self._run_chain(arr, progress_cb)

        # Pure mono 2D
        if arr.ndim == 2:
            out = self._run_chain(arr, progress_cb)
            return np.stack([out, out, out], axis=2)

        # Fallback
        out = self._run_chain(arr, progress_cb)
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=2)
        return out

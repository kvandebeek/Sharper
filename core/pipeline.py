from workflows.preblur import PreBlur
from workflows.preprocess import Preprocess
from workflows.deconvolution import Deconvolution
from workflows.wavelets import Wavelets
from workflows.detail_boost import DetailBoost
from workflows.noise_reduction import NoiseReduction
from workflows.final_adjust import FinalAdjust


class Pipeline:
    def __init__(self):
        dummy_ref = [None]  # kept to satisfy workflow constructors, but not used
        self.items = [
            PreBlur(dummy_ref),         # 0
            Preprocess(dummy_ref),      # 1 (currently identity)
            Deconvolution(dummy_ref),   # 2
            Wavelets(dummy_ref),        # 3
            DetailBoost(dummy_ref),     # 4
            NoiseReduction(dummy_ref),  # 5
            FinalAdjust(dummy_ref)      # 6
        ]

    def apply_all(self, img_rgb):
        out = img_rgb.astype("float32")
        maxv = float(out.max()) if out.size else 1.0
        if maxv > 1.5:
            out /= 255.0
        for item in self.items:
            out = item.apply(out)
        return out.clip(0.0, 1.0)

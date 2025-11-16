import numpy as np
from workflows.preblur import PreBlur
from workflows.preprocess import Preprocess
from workflows.deconvolution import Decon
from workflows.wavelets import Wavelets
from workflows.detail import Detail
from workflows.denoise import Denoise
from workflows.final import Final

class Pipeline:
    def __init__(self):
        self.items = [
            PreBlur(),      # 0
            Preprocess(),   # 1
            Decon(),        # 2
            Wavelets(),     # 3
            Detail(),       # 4
            Denoise(),      # 5
            Final()         # 6
        ]

    def apply_all(self, rgb, progress=None):
        x = rgb.astype(np.float32) / 255.0
        total = len(self.items)
        for i, step in enumerate(self.items):
            if progress: progress(int(i/total*100), step.name)
            x = step.process(x)
        if progress: progress(100, "Done")
        return x

    def reset_all(self):
        pass

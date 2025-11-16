import numpy as np, cv2
from dataclasses import dataclass

@dataclass
class Decon:
    name: str = "Deconvolution"
    iterations: int = 0
    psf_sigma: float = 1.5

    def process(self, img):
        if self.iterations <= 0: return img
        s = max(0.3, self.psf_sigma)
        r = int(s*6); r += 1 - r%2
        k1d = cv2.getGaussianKernel(r, s)
        psf = k1d @ k1d.T
        psf /= psf.sum()
        psf_m = psf[::-1, ::-1]

        est = img.copy()
        for _ in range(self.iterations):
            conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
            ratio = img / (conv + 1e-6)
            est *= cv2.filter2D(ratio, -1, psf_m, borderType=cv2.BORDER_REFLECT)

        return np.clip(est,0,1)

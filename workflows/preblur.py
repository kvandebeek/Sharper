import cv2
from dataclasses import dataclass

@dataclass
class PreBlur:
    name: str = "PreBlur"
    sigma: float = 0.0
    def process(self, img):
        if self.sigma <= 0: return img
        s = max(0.3, self.sigma)
        k = int(s*6); k += 1 - k%2
        return cv2.GaussianBlur(img, (k,k), s)

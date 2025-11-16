import cv2
from dataclasses import dataclass

@dataclass
class Denoise:
    name: str = "Denoise"
    strength: float = 0.0
    radius: float = 1.0

    def process(self,img):
        if self.strength<=0: return img
        d = max(5, int(self.radius*10))
        sigma_c = self.strength*25
        sigma_s = self.radius*25
        out = cv2.bilateralFilter(img,d,sigma_c,sigma_s)
        return out

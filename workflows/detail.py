import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class Detail:
    name: str = "Detail"
    amount: float = 0.0
    radius: float = 2.0
    def process(self, img):
        if self.amount <= 0: return img
        r = max(0.3, self.radius)
        k = int(r*6); k += 1 - k%2
        blur = cv2.GaussianBlur(img,(k,k),r)
        return np.clip(img + (img-blur)*self.amount,0,1)

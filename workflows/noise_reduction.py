import cv2
import numpy as np
from core.workflow_item import WorkflowItem

class NoiseReduction(WorkflowItem):
    def __init__(self, image_ref, strength=0.0, radius=1.0):
        super().__init__("NoiseReduction")
        self.image_ref = image_ref
        self.strength = strength  # 0 = off
        self.radius = radius

    def _process_channel(self, ch):
        ch = ch.astype(np.float32)
        if self.strength <= 0.0:
            return ch
        temp = (ch * 255.0).astype(np.float32)
        try:
            filt = cv2.bilateralFilter(
                temp,
                d=0,
                sigmaColor=float(self.strength),
                sigmaSpace=float(self.radius)
            )
        except Exception:
            return ch
        out = filt / 255.0
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return out.clip(0.0, 1.0)

    def apply(self, img):
        img = img.astype(np.float32)
        if img.ndim == 2:
            return self._process_channel(img)
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = self._process_channel(img[..., c])
        return out

    def execute(self):
        self.image_ref[0] = self.apply(self.image_ref[0])

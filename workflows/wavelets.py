import numpy as np
import cv2
from core.workflow_item import WorkflowItem

class Wavelets(WorkflowItem):
    def __init__(self, image_ref, gains=None):
        super().__init__("Wavelets")
        self.image_ref = image_ref
        self.gains = gains or [0.0, 0.0, 0.0, 0.0, 0.0]  # all 0 = off

    def _process_channel(self, ch):
        ch = ch.astype(np.float32)
        if not any(self.gains):
            return ch
        gauss = [ch]
        for i in range(1, 6):
            sigma = 1.2 * i
            gauss.append(cv2.GaussianBlur(ch, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101))
        details = [gauss[i] - gauss[i+1] for i in range(5)]
        out = gauss[-1].copy()
        for i in range(5):
            out += details[i] * float(self.gains[i])
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

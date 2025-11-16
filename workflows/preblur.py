import cv2
import numpy as np
from core.workflow_item import WorkflowItem

class PreBlur(WorkflowItem):
    def __init__(self, image_ref, sigma=0.0):
        super().__init__("PreBlur")
        self.image_ref = image_ref
        self.sigma = sigma  # 0 = off

    def _blur_channel(self, ch):
        if self.sigma <= 0.0:
            return ch
        blurred = cv2.GaussianBlur(ch, (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT_101)
        blurred = np.nan_to_num(blurred, nan=0.0, posinf=1.0, neginf=0.0)
        return blurred.clip(0.0, 1.0)

    def apply(self, img):
        img = img.astype(np.float32)
        if img.ndim == 2:
            return self._blur_channel(img)
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = self._blur_channel(img[..., c])
        return out

    def execute(self):
        self.image_ref[0] = self.apply(self.image_ref[0])

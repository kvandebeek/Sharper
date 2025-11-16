import cv2
import numpy as np
from core.workflow_item import WorkflowItem

class DetailBoost(WorkflowItem):
    def __init__(self, image_ref, amount=0.0, radius=2.0):
        super().__init__("DetailBoost")
        self.image_ref = image_ref
        self.amount = amount    # 0 = off
        self.radius = radius

    def _process_channel(self, ch):
        ch = ch.astype(np.float32)
        if self.amount == 0.0:
            return ch
        blur = cv2.GaussianBlur(ch, (0, 0), self.radius, borderType=cv2.BORDER_REFLECT_101)
        high = ch - blur
        out = ch + high * float(self.amount)
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

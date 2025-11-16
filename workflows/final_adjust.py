import numpy as np
from core.workflow_item import WorkflowItem

class FinalAdjust(WorkflowItem):
    def __init__(self, image_ref, gamma=1.0, contrast=1.0, brightness=0.0):
        super().__init__("FinalAdjust")
        self.image_ref = image_ref
        self.gamma = gamma          # 0.3..3
        self.contrast = contrast    # 0.3..3
        self.brightness = brightness # -0.5..+0.5

    def _process_channel(self, ch):
        out = ch.astype(np.float32)
        out = out + float(self.brightness)
        out = (out - 0.5) * float(self.contrast) + 0.5
        out = out.clip(0.0, 1.0)
        if self.gamma != 1.0:
            out = np.power(out, 1.0 / float(self.gamma))
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

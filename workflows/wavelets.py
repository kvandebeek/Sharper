from core.workflow_item import WorkflowItem
import numpy as np
import cv2

class Wavelets(WorkflowItem):
    def __init__(self, image_ref, gains=None):
        super().__init__("Wavelets")
        self.image_ref = image_ref
        self.gains = gains or [0, 0, 0, 0, 0]

        # MUCH better sigmas
        self.sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]

    def _process_channel(self, ch):
        ch = ch.astype(np.float32)

        # compute blur levels
        gauss = [ch]
        for sigma in self.sigmas:
            gauss.append(
                cv2.GaussianBlur(ch, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
            )

        # compute details
        details = [gauss[i] - gauss[i+1] for i in range(len(self.sigmas))]

        # base = ORIGINAL IMAGE, not gauss[-1]
        out = ch.copy()

        # apply weighted details
        for i in range(len(self.sigmas)):
            out += details[i] * float(self.gains[i])

        return np.clip(np.nan_to_num(out), 0.0, 1.0)

    def apply(self, img):
        if img.ndim == 2:
            return self._process_channel(img)

        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = self._process_channel(img[..., c])
        return out

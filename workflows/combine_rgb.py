import cv2
import numpy as np
from core.workflow_item import WorkflowItem

class FinalCombineRGB(WorkflowItem):
    def __init__(self, image_ref):
        super().__init__("CombineRGB")
        self.image_ref = image_ref
        self.lab_src = None  # window will set this

    def combine(self, lab_src):
        L_new = self.image_ref[0]  # 0..1 float
        lab = lab_src.copy()
        lab[...,0] = L_new * 100.0
        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        return np.clip(rgb, 0.0, 1.0)

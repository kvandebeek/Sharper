import numpy as np
from core.workflow_item import WorkflowItem

class Preprocess(WorkflowItem):
    def __init__(self, image_ref):
        super().__init__("Preprocess")
        self.image_ref = image_ref

    def apply(self, img):
        return img.astype(np.float32)

    def execute(self):
        self.image_ref[0] = self.apply(self.image_ref[0])

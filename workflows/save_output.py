import numpy as np
from PIL import Image
from core.workflow_item import WorkflowItem

class SaveOutput(WorkflowItem):
    def __init__(self, image_ref, path="output.png", bitdepth=16):
        super().__init__("Save Output")
        self.image_ref = image_ref
        self.path = path
        self.bitdepth = bitdepth  # 8 or 16

    def set_path(self, path):
        self.path = path

    def execute(self):
        img = self.image_ref[0]
        if img is None:
            print("No image to save")
            return

        if self.bitdepth == 8:
            arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            arr = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
            pil = Image.fromarray(arr)

        pil.save(self.path)
        print(f"Saved output image to {self.path}")

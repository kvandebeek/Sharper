import cv2
from core.workflow_item import WorkflowItem

class LoadImage(WorkflowItem):
    def __init__(self):
        super().__init__("LoadImage")
        self.filename = None

    def set_file(self, filename):
        self.filename = filename
        self.mark_dirty()

    def process(self, img_in, **params):
        if not self.filename:
            return None
        return cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)

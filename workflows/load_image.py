from core.workflow_item import WorkflowItem
from processing.loader import ImageLoader

class LoadImage(WorkflowItem):
    def __init__(self, path=None):
        super().__init__("Load Image")
        self.path = path
        self.image = None

    def set_path(self, path):
        self.path = path

    def execute(self):
        if not self.path:
            print("No image path set")
            return

        loader = ImageLoader()
        self.image = loader.load_one(self.path)
        print(f"Loaded image: {self.path}")

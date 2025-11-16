from core.pipeline import Pipeline
from ui.window import start_qt

class App:
    def __init__(self):
        # The pipeline builds itself internally
        self.pipeline = Pipeline()

    def run(self):
        start_qt(self.pipeline)

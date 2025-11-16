from core.pipeline_step import PipelineStep

class Contrast(PipelineStep):
    def __init__(self, name="Contrast"):
        super().__init__(name)
        self.contrast = 1.0   # UI expects .contrast
        self.brightness = 0.0 # if UI uses it later

    def process(self, img):
        out = img * self.contrast + self.brightness
        return out

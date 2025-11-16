class PipelineStep:
    def __init__(self, name):
        self.name = name
        self.prev = None
        self.next = None
        self.dirty = True
        self.cached = None

    def mark_dirty(self):
        if not self.dirty:
            self.dirty = True
            if self.next:
                self.next.mark_dirty()

    def run(self, img):
        if not self.dirty and self.cached is not None:
            return self.cached

        if self.prev:
            img = self.prev.run(img)

        out = self.process(img)
        self.cached = out
        self.dirty = False
        return out

    def process(self, img):
        return img

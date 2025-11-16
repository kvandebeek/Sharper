class WorkflowItem:
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.cached = None
        self.dirty = True

        self.input_item = None
        self.output_item = None

    def set_params(self, **kwargs):
        if kwargs != self.params:
            self.params = kwargs
            self.mark_dirty()

    def mark_dirty(self):
        if not self.dirty:
            self.dirty = True
            if self.output_item:
                self.output_item.mark_dirty()

    def process_cached(self):
        if not self.dirty:
            return self.cached

        img_in = None
        if self.input_item:
            img_in = self.input_item.process_cached()

        self.cached = self.process(img_in, **self.params)
        self.dirty = False
        return self.cached

    def process(self, img_in, **params):
        raise NotImplementedError

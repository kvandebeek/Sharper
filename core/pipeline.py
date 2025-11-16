class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        self._link()

    def _link(self):
        for i in range(len(self.steps) - 1):
            a = self.steps[i]
            b = self.steps[i + 1]
            a.output_item = b
            b.input_item = a

    def run(self):
        return self.steps[-1].process_cached()

    def mark_all_dirty(self):
        for step in self.steps:
            step.dirty = True
            step.cached = None

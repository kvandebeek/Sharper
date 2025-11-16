class WorkflowPanel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def render(self):
        print("WorkflowPanel:")
        for item in self.pipeline.items:
            print(f" - {item.name}")

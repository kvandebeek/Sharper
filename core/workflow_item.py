class WorkflowItem:
    def __init__(self, name: str):
        self.name = name

    def execute(self):
        raise NotImplementedError("WorkflowItem subclasses must implement execute()")

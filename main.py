from core.app import Pipeline
from ui.window import start_qt

if __name__ == "__main__":
    pipeline = Pipeline()
    start_qt(pipeline)

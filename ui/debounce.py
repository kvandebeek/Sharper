from PySide6.QtCore import QTimer

class Debounce:
    def __init__(self, delay_ms=300):
        self.timer = QTimer()
        self.timer.setInterval(delay_ms)
        self.timer.setSingleShot(True)
        self.func = None

        self.timer.timeout.connect(self._run)

    def call(self, func):
        self.func = func
        self.timer.start()

    def _run(self):
        if self.func:
            self.func()

from dataclasses import dataclass
@dataclass
class Preprocess:
    name: str = "Preprocess"
    def process(self, img): return img

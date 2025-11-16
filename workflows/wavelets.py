from dataclasses import dataclass, field
from workflows.wavelet_atrous import atrous_wavelet

@dataclass
class Wavelets:
    name: str = "Wavelets"
    gains: list = field(default_factory=lambda:[0,0,0,0,0])
    def process(self, img): return atrous_wavelet(img, self.gains)

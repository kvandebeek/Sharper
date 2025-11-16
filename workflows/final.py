import numpy as np
from dataclasses import dataclass

@dataclass
class Final:
    name: str = "Final"
    gamma: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    def process(self,img):
        out = img.astype(np.float32)
        if abs(self.gamma-1)>1e-3: out = out**(1.0/self.gamma)
        if abs(self.contrast-1)>1e-3: out = 0.5+(out-0.5)*self.contrast
        if abs(self.brightness)>1e-4: out += self.brightness
        return np.clip(out,0,1)

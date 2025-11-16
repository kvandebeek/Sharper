from PIL import Image
import numpy as np

class ImageLoader:
    def load_one(self, path):
        img = Image.open(path)

        # --- Fix alpha issues (macOS PNG often returns BGRA) ---
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Grayscale with alpha → RGB
        elif img.mode == "LA":
            img = img.convert("L").convert("RGB")

        # Indexed, CMYK → RGB
        elif img.mode in ("P", "CMYK"):
            img = img.convert("RGB")

        # At this point we guarantee RGB 3-channel image
        arr = np.array(img)

        return arr

    def load_many(self, paths):
        return [self.load_one(p) for p in paths]

import json
import os

class PresetManager:
    def __init__(self, preset_dir):
        self.preset_dir = preset_dir

    def list_presets(self):
        return [f[:-5] for f in os.listdir(self.preset_dir) if f.endswith(".json")]

    def load(self, name):
        path = os.path.join(self.preset_dir, name + ".json")
        with open(path, "r") as f:
            return json.load(f)

    def save(self, name, data):
        path = os.path.join(self.preset_dir, name + ".json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

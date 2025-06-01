import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ETLDataset(Dataset):
    def __init__(self, root_dir, etl_dirs, label_map_path, transform=None):
        # Load label_map: index (str) -> code_dir (e.g., '0x8702')
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        # Invert map: code_dir -> index (str)
        self.code_to_index = {code.upper(): idx for idx, code in label_map.items()}
        self.samples = []
        for etl in etl_dirs:
            etl_path = os.path.join(root_dir, etl)
            if not os.path.isdir(etl_path):
                continue
            for code_dir in os.listdir(etl_path):
                code_key = code_dir.upper()
                if code_key not in self.code_to_index:
                    continue
                idx = int(self.code_to_index[code_key])
                full_dir = os.path.join(etl_path, code_dir)
                if not os.path.isdir(full_dir):
                    continue
                for fn in os.listdir(full_dir):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(full_dir, fn), idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label
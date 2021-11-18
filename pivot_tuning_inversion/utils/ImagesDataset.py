import os

from torch.utils.data import Dataset
from PIL import Image

from pivot_tuning_inversion.utils.data_utils import make_dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root, device, source_transform=None):
        if isinstance(source_root, list):
            self.pil_input = True
        else:
            self.pil_input = False
            self.source_paths = sorted(make_dataset(source_root))
        self.source_root = source_root
        self.device = device
        self.source_transform = source_transform

    def __len__(self):
        if self.pil_input:
            return len(self.source_root)
        else:
            return len(self.source_paths)

    def __getitem__(self, index):
        if self.pil_input:
            fname = str(index)
            from_im = self.source_root[index]
        else:
            fname, from_path = self.source_paths[index]
            from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)

        return fname, from_im.to(self.device)

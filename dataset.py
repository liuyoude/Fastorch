import torch
from torch.utils.data import Dataset

import utils


class MyDataset(Dataset):
    def __init__(self, dirs: list):
        self.filename_list = []
        for dir in dirs:
            self.filename_list.extend(utils.get_filename_list(dir))

    def __getitem__(self, item):
        filename = self.filename_list[item]
        return self.transform(filename)

    def transform(self, filename):
        return filename

    def __len__(self):
        return len(self.filename_list)
import torch
from torch.utils.data import Dataset

import utils


class MyDataset(Dataset):
    def __init__(self, dirs: list, load_in_memory=False):
        self.file_list = []
        self.load_in_memory = load_in_memory
        for dir in dirs:
            filename_list = utils.get_filename_list(dir)
            if load_in_memory:
                # speed for training
                self.file_list.extend([self.transform(filename) for filename in filename_list])
            else:
                self.filename_list.extend(utils.get_filename_list(dir))

    def __getitem__(self, item):
        data_item = self.file_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    @classmethod
    def transform(cls, filename):
        # preprocess (default Identity)
        data = filename
        return data

    def __len__(self):
        return len(self.file_list)

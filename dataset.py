import torch
from torch.utils.data import Dataset

import utils


class MyDataset(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.args = args
        self.file_list = file_list
        self.load_in_memory = load_in_memory
        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    def transform(self, filename):
        # preprocess (default Identity)
        data = filename
        return data

    def __len__(self):
        return len(self.file_list)

import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups

NEWSGROUPS_DIR = '../data/20news'

class NewsGroupsDataset(Dataset):
    def __init__(self, dir = NEWSGROUPS_DIR, split = 'train'):
        super().__init__()

        self.dataset = fetch_20newsgroups(subset=split, data_home=dir, download_if_missing=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset['data'][idx], self.dataset['target'][idx]

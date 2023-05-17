from xml.etree import ElementTree
import torch
from torch.utils.data import Dataset
import os
from os.path import isdir

NYTIMES_DIR = '../data/nyt_corpus'

class NewsGroupsDataset(Dataset):
    def __init__(self, dir=NYTIMES_DIR, split='train'):
        super().__init__()

        self.load_data(dir, range(2000, 2001))

    def load_data(self, dir, years : [int]):
        self.dataset = []
        for year in years:
            for month in range(1,13):
                for day in range(1, 33):
                    fdir = f'{dir}/{year}/{month:02}/{day:02}'
                    if not isdir(fdir):
                        continue

                    self.dataset.extend([f'{fdir}/{xml}' for xml in os.listdir(fdir)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tree = ElementTree.parse(self.dataset[idx]).getroot()
        s = ''
        for i, element in enumerate(tree.iter()):
            if element.text is None:
                continue
            element_text = element.text.strip()

            s += element_text + '\n'

        return s
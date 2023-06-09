from xml.etree import ElementTree
import torch
from torch.utils.data import Dataset
import os
from os.path import isdir

NYTIMES_DIR = '../data/nyt_corpus'

class NYTimesDataset(Dataset):
    def __init__(self, tokenizer, out_dim, data_dir=NYTIMES_DIR, split='train'):
        super().__init__()

        self.load_data(data_dir, range(2000, 2001))
        self.tokenizer = tokenizer
        self.out_dim = out_dim

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

        return self.process_line(s)

    def process_line(self, line):

        content = line.strip()
        # content = line.split("\t")

        encoded = self.tokenizer(
            f"{content}",
            add_special_tokens=True,
            max_length=self.out_dim,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        target_encoded = self.tokenizer(
            f"{content}",
            add_special_tokens=True,
            max_length=self.out_dim,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        encoder_input = encoded.input_ids.squeeze(0)
        encoder_mask = encoded.attention_mask.squeeze(0)
        decoder_target = target_encoded.input_ids.squeeze(0)

        return (
            encoder_input,
            encoder_mask,
            decoder_target,
        )
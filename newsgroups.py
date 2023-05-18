import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups

NEWSGROUPS_DIR = '../data/20news'

class NewsGroupsDataset(Dataset):
    def __init__(self, tokenizer, out_dim, data_dir = NEWSGROUPS_DIR, split ='train'):
        super().__init__()
        if split == 'val':
            split = 'test'

        self.dataset = fetch_20newsgroups(subset=split, data_home=data_dir, download_if_missing=False)
        self.tokenizer = tokenizer
        self.out_dim = out_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.process_line(self.dataset['data'][idx])

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
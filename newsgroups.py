import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
import pickle as pkl
import numpy as np
from transformers import T5TokenizerFast

NEWSGROUPS_DIR = '../data/20news'

class NewsGroupsDataset(Dataset):
    def __init__(self, tokenizer : T5TokenizerFast, out_dim, data_dir = NEWSGROUPS_DIR, split ='train'):
        super().__init__()
        if split == 'val':
            split = 'test'

        self.dataset = fetch_20newsgroups(subset=split, data_home=data_dir, download_if_missing=False)
        self.tokenizer = tokenizer
        self.out_dim = out_dim
        self.vocab = self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.dataset['data'])

    def get_bow_from_sentence(self, tokenized_sentence):
        bow_vector = torch.zeros((len(self.vocab)), dtype=torch.float32)
        # get bow represetentation
        for tensor_token in tokenized_sentence:
            bow_vector[tensor_token] += 1

        return bow_vector


    def __getitem__(self, idx):
        line = self.dataset['data'][idx]

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

        all_encoded = self.tokenizer(
            f"{content}",
            add_special_tokens=True,
            max_length=None,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        bow_from_sentence = self.get_bow_from_sentence(all_encoded.input_ids.squeeze(0))

        return (
            encoder_input,
            encoder_mask,
            decoder_target,
            bow_from_sentence / bow_from_sentence.sum(), # get distribution of words (probabilities)
        )
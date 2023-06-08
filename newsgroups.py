import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
import pickle as pkl
import numpy as np
from transformers import T5TokenizerFast
from gensim.corpora import Dictionary
import re

NEWSGROUPS_DIR = '../data/20news'

class NewsGroupsDataset(Dataset):
    def __init__(self, tokenizer : T5TokenizerFast, out_dim, data_dir = NEWSGROUPS_DIR, split ='train'):
        super().__init__()
        if split == 'val':
            split = 'test'

        self.dataset = fetch_20newsgroups(subset=split, data_home=data_dir, download_if_missing=False)
        self.tokenizer = tokenizer
        self.out_dim = out_dim
        self.dictionary : Dictionary = Dictionary.load(f'{NEWSGROUPS_DIR}/20news_vocab.dict')

    def __len__(self):
        return len(self.dataset['data'])

    def get_vocab_size(self):
        return len(self.dictionary)

    def get_vocabulary(self):
        return self.dictionary

    def get_bow_from_sentence(self, document):
        tokenizer = lambda s: re.findall('\w+', s.lower())
        tokens = tokenizer(document)

        bow_representation = self.dictionary.doc2bow(tokens)

        bow_tensor = torch.zeros(len(self.dictionary))

        for token_id, token_count in bow_representation:
            bow_tensor[token_id] = token_count

        return bow_tensor

    def __getitem__(self, idx):
        return self.process_document(self.dataset['data'][idx])

    def process_document(self, document):
        content = document.strip()
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

        bow_from_sentence = self.get_bow_from_sentence(content)

        return (
            encoder_input,
            encoder_mask,
            decoder_target,
            bow_from_sentence / bow_from_sentence.sum(),  # get distribution of words (probabilities)
        )

    def get_documents_by_topic(self) -> [[str]]:
        # returns a list of docs by topics
        curr_target = 0
        all_list = [[]]

        for data, target in sorted(zip(self.dataset['data'], self.dataset['target']), key=lambda t : t[1]):
            if target != curr_target:
                all_list.append([])
                curr_target = target

            all_list[-1].append(data)

        return all_list
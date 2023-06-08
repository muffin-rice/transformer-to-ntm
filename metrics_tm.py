from transformer2vae.model_t5 import T5VAE
from transformers import T5TokenizerFast
import numpy as np
import torch
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def get_keywords_from_docs(model : T5VAE, vocab : Dictionary, documents_encoded : [], document_masks : [], top_n = 10) -> [int]:
    bows = []
    for doc, doc_mask in zip(documents_encoded, document_masks):
        _, _, _, _, bow = model.forward(doc, doc_mask, None)
        bows.append(bow.to_numpy())

    bows = torch.Tensor(bows).sum(dim=0) # sum across docs
    argsorted = torch.argsort(bows, descending=True)

    # in the argsorts, find if any args are stopwords and ignore them
    stops = set(stopwords.words('english'))

    top_bow = []
    for arg in argsorted:
        if len(top_bow) >= top_n:
            break

        word = vocab[arg.item()]
        # check if word is stop

        if word in stops:
            continue

        top_bow.append(word)

    return top_bow


def calculate_coherences(topics : [str], documents : [], vocab : Dictionary):
    '''
    :param topics:
    :param documents:
    :param vocab:
    :return:c_npmi
    '''
    cm = CoherenceModel(topics = topics,
                        texts = documents,
                        coherence='c_npmi',
                        dictionary = vocab)

    coherence_per_topic = cm.get_coherence_per_topic()

    return coherence_per_topic

def calc_coherence_newsgroups(model, dataset):
    # from newsgroups import NewsGroupsDataset

    tokenizer = T5TokenizerFast.from_pretrained('t5-small')

    # dataset = NewsGroupsDataset(tokenizer, out_dim=32)

    docs_by_topic = dataset.get_documents_by_topic()

    vocabulary = dataset.get_vocabulary()

    keywords_topics = []

    for docs in docs_by_topic:
        documents_encoded = []
        documents_masks = []
        for doc in docs:
            inp, mask, _, _ = dataset.process_document(doc)
            documents_encoded.append(inp)
            documents_masks.append(mask)

        keywords_topics.append(get_keywords_from_docs(model, vocabulary, documents_encoded=documents_encoded,
                                                      document_masks = documents_masks))

    npmis = []
    for generated_topics, documents in zip(keywords_topics, docs_by_topic):
        npmis.append(calculate_coherences(generated_topics, documents, vocabulary))

    return npmis

if __name__ == '__main__':
    pass

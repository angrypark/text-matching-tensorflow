# -*- coding: utf-8 -*-
from collections import Counter
from tqdm import tqdm
from gensim.models import FastText
import os

from utils.utils import JamoProcessor

class Vectorizer:
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.pretrained_embed_dir = config.pretrained_embed_dir
        self.vocab_list = config.vocab_list
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.idx2word = list()
        self.word2idx = dict()
        self.fasttext = FastText()
        
    def build_vectorizer(self):
        if self.vocab_list:
            self.word2idx, self.idx2word = self._load_vocab()
        else:
            self.word2idx, self.idx2word = self._build_vocab()

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except KeyError:
            return self.word2idx[self.UNK_TOKEN]

    def _build_vocab(self):
        count = Counter()
        processor = JamoProcessor()
        self.fasttext = FastText.load(self.pretrained_embed_dir)
        fname = os.listdir(self.base_dir)[0]
        with open("/media/scatter/scatterdisk/reply_matching_model/sol.preprocessed_1.txt", "r") as f:
            for line in f:
                corpus_id, query, reply = line.strip().split("\t")
                count.update(self.tokenizer.tokenize(query))
                count.update(self.tokenizer.tokenize(reply))
        idx2word = [self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN] +  \
                    sorted([word for word, _ in count.most_common(self.vocab_size-3)])
        word2idx = {word:idx for idx, word in enumerate(idx2word)}
        return word2idx, idx2word
    
    def _load_vocab(self):
        with open(self.vocab_list, "r") as f:
            idx2word = [line.strip() for line in f if line.strip()]
        word2idx = {word:idx for idx, word in enumerate(idx2word)}
        return word2idx, idx2word

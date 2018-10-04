import tensorflow as tf
import numpy as np
import os
import sys
from collections import namedtuple
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import editdistance

from text.tokenizers import SentencePieceTokenizer
from utils.utils import JamoProcessor

Config = namedtuple("config", ["sent_piece_model"])
processor = JamoProcessor()
tokenizer = SentencePieceTokenizer(Config("/media/scatter/scatterdisk/tokenizer/sent_piece.100K.model"))

def my_word_tokenizer(raw, pos=[], stopword=[]):
    return tokenizer.tokenize(raw)

def my_char_tokenizer(raw, pos=[], stopword=[]):
    return [processor.word_to_jamo(word) for word in tokenizer.tokenize(raw)]

def edit_distance_ratio(a_jamos, b_jamos):
    long_length = max([len(a_jamos), len(b_jamos)])
    ratio = editdistance.eval(a_jamos, b_jamos) / long_length
    return ratio

class FeatureExtractor:
    def __init__(self, tfidf_char_vectorizer_dir=None, ):
        self.tfidf_char_vectorizer = pickle.load(open(tfidf_char_vectorizer_dir, "rb"))
        self.tfidf_word_vectorizer = None

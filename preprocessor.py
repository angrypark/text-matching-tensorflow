import numpy as np
from text import normalizers, tokenizers, vectorizers

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen=np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


class Preprocessor:
    def __init__(self, config):
        self.min_length = config.min_length
        self.max_length = config.max_length
        self.normalizer = getattr(normalizers, config.normalizer)()
        self.tokenizer = getattr(tokenizers, config.tokenizer)(config)
        self.vectorizer = vectorizers.Vectorizer(self.tokenizer, config)
        self.feature_extractor = None
        self.build_preprocessor()

    def build_preprocessor(self):
        self.vectorizer.build_vectorizer()

    def _preprocess(self, sentence):
        normalized_sentence = self.normalizer.normalize(sentence)
        tokenized_sentence = ["<SOS>"] + self.tokenizer.tokenize(normalized_sentence) + ["<EOS>"]
        indexed_sentence = [self.vectorizer.indexer(token) for token in tokenized_sentence]
        return indexed_sentence, len(indexed_sentence)

    def preprocess(self, sentence):
        indexed_sentence, length = self._preprocess(sentence)
        padded_sentence = pad_sequences([indexed_sentence], maxlen=self.max_length)[0]
        return padded_sentence, length


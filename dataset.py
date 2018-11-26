import numpy as np
from collections import defaultdict
import tensorflow as tf
import os


class DummyDataset:
    """
    Just for inference. Imports nothing, reads nothing
    """
    
    def __init__(self):
        pass
            
    def get_data_iterator(self, index_table, mode=tf.contrib.learn.ModeKeys.TRAIN):
        return self
    
    def get_next(self):
        return defaultdict(list)

class Dataset:
    def __init__(self, preprocessor, train_dir, val_dir, min_length, max_length,
                 num_negative_samples, batch_size, shuffle, num_epochs, debug=False):
        self.preprocessor = preprocessor
        self.val_fnames = self.get_fnames(val_dir)
        self.train_fnames = [fname for fname in self.get_fnames(train_dir) if fname not in self.val_fnames]
        self.min_length = min_length
        self.max_length = max_length
        self.num_negative_samples = num_negative_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs

        # Debug mode
        if debug:
            self.train_size = 1000
            self.val_size = 1000
        # Train mode
        else:
            self.train_size = 298554955
            self.val_size = 219686
            
    def get_data_iterator(self, index_table, mode=tf.contrib.learn.ModeKeys.TRAIN):
        if mode==tf.contrib.learn.ModeKeys.TRAIN:
            train_set = tf.data.TextLineDataset(self.train_fnames)
            train_set = train_set.map(lambda line: self.parse_single_line_old(line, index_table, self.max_length), 
                                      num_parallel_calls=10)
            train_set = train_set.shuffle(buffer_size=500)
            train_set = train_set.batch(self.batch_size)
            train_set = train_set.prefetch(1)
            train_set = train_set.repeat(self.num_epochs)

            train_iterator = train_set.make_initializable_iterator("train_iterator")
            return train_iterator

        elif mode==tf.contrib.learn.ModeKeys.EVAL:
            val_set = tf.data.TextLineDataset(self.val_fnames)
            val_set = val_set.map(lambda line: self.parse_single_line_old(line, index_table, self.max_length),
                                  num_parallel_calls=10)
            val_set = val_set.shuffle(buffer_size=500)
            val_set = val_set.batch(self.batch_size)

            val_iterator = val_set.make_initializable_iterator("val_iterator")
            return val_iterator

        else:
            raise NotImplementedError

    def parse_single_line(self, line, index_table):
        """
        Preprocess one line.
        It assumes that train set is normalized, tokenized.
        With given index table, it converts a sentence to list of indices.
        """
        splited = tf.string_split([line], delimiter="\t")
        query = tf.concat([["<SOS>"], tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]], axis=0)
        reply = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0)
        query = tf.squeeze(index_table.lookup(query))
        reply = tf.squeeze(index_table.lookup(reply))

        return {"input_queries": query,
                "input_replies": reply,
                "query_lengths": [tf.shape(query)[0]],
                "reply_lengths": [tf.shape(reply)[0]]}

    def get_fnames(self, dir):
        if os.path.isdir(dir):
            return [os.path.join(dir, fname) for fname in sorted(os.listdir(dir))]
        elif os.path.isfile(dir):
            return [dir]
        else:
            raise FileNotFoundError

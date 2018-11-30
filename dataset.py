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
    def __init__(self, preprocessor, train_dir, val_dir, min_length, max_length, num_negative_samples, batch_size, shuffle, num_epochs, debug=False):
        self.preprocessor = preprocessor
        self.val_fnames = self.get_fnames(val_dir)
        self.train_fnames = [fname for fname in self.get_fnames(train_dir) if fname not in self.val_fnames]
        self.min_length = min_length
        self.max_length = max_length
        self.num_negative_samples = num_negative_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.pad_shapes = {"input_queries": [None],
                           "input_replies": [None],
                           "query_lengths": [None],
                           "reply_lengths": [None]}
        if debug:
            self.train_size = 1000
            self.val_size = 1000
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
    
    def get_negatives_iterator(self, index_table, mode=tf.contrib.learn.ModeKeys.TRAIN):
        if mode==tf.contrib.learn.ModeKeys.TRAIN:
            train_set = tf.data.TextLineDataset(self.train_fnames)
            train_set = train_set.map(lambda line: self.parse_single_negative(line, index_table, self.max_length), 
                                      num_parallel_calls=1)
            train_set = train_set.shuffle(buffer_size=2, seed=777)
            train_set = train_set.batch(self.batch_size*self.num_negative_samples)
            train_set = train_set.prefetch(1)
            train_set = train_set.repeat(self.num_epochs*self.num_negative_samples+1)

            train_iterator = train_set.make_initializable_iterator("train_negatives_iterator")
            return train_iterator

        elif mode==tf.contrib.learn.ModeKeys.EVAL:
            val_set = tf.data.TextLineDataset(self.val_fnames)
            val_set = val_set.map(lambda line: self.parse_single_negative(line, index_table, self.max_length),
                                  num_parallel_calls=10)
            val_set = val_set.shuffle(buffer_size=10, seed=777)
            val_set = val_set.batch(self.batch_size)

            val_iterator = val_set.make_initializable_iterator("val_negatives_iterator")
            return val_iterator

        else:
            raise NotImplementedError

    def parse_single_line(self, line, index_table):
        splited = tf.string_split([line], delimiter="\t")
        query = tf.concat([["<SOS>"], tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]], axis=0)
        reply = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0)
        query = tf.squeeze(index_table.lookup(query))
        reply = tf.squeeze(index_table.lookup(reply))

        return {"input_queries": query,
                "input_replies": reply,
                "query_lengths": [tf.shape(query)[0]],
                "reply_lengths": [tf.shape(reply)[0]]}
    
    def parse_single_line_old(self, line, index_table, max_length):
        """get single line from train set, and returns after padding and indexing
        :param line: corpus id \t query \t reply
        """
        splited = tf.string_split([line], delimiter="\t")
        query = tf.concat([["<SOS>"], tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]], axis=0)
        reply = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0)
        query_length = tf.minimum(tf.shape(query)[0], max_length)
        reply_length = tf.minimum(tf.shape(reply)[0], max_length)
        
        paddings = tf.constant([[0, 0],[0, max_length]])
        padded_query = tf.slice(tf.pad([query], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
        padded_reply = tf.slice(tf.pad([reply], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
        
        indexed_query = tf.squeeze(index_table.lookup(padded_query))
        indexed_reply = tf.squeeze(index_table.lookup(padded_reply))

        return {"input_queries": indexed_query,
                "input_replies": indexed_reply,
                "query_lengths": query_length,
                "reply_lengths": reply_length}
    
    def parse_single_negative(self, line, index_table, max_length):
        """get single line from train set, and returns after padding and indexing
        :param line: corpus id \t query \t reply
        """
        splited = tf.string_split([line], delimiter="\t")
        reply = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0)
        reply_length = tf.shape(reply)[0]
        
        paddings = tf.constant([[0, 0],[0, max_length]])
        padded_reply = tf.slice(tf.pad([reply], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
        
        indexed_reply = tf.squeeze(index_table.lookup(padded_reply))

        return {"input_negatives": indexed_reply,
                "negative_lengths": reply_length}
                                 
    def get_fnames(self, dir):
        if os.path.isdir(dir):
            return [os.path.join(dir, fname) for fname in sorted(os.listdir(dir)) if "vocab" not in fname]
        elif os.path.isfile(dir):
            return [dir]
        else:
            raise FileNotFoundError

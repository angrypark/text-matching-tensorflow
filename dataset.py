import numpy as np
import tensorflow as tf
import os

class Dataset:
    def __init__(self, preprocessor, train_dir, val_dir, min_length, max_length, batch_size, shuffle, num_epochs):
        self.preprocessor = preprocessor
        self.val_fnames = self.get_fnames(val_dir)
        self.train_fnames = [fname for fname in self.get_fnames(train_dir) if fname not in self.val_fnames]
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def get_data_iterator(self, index_table, mode=tf.contrib.learn.ModeKeys.TRAIN):
        if mode==tf.contrib.learn.ModeKeys.TRAIN:
            train_set = tf.data.TextLineDataset(self.train_fnames)
            train_set = train_set.map(lambda line: self.parse_single_line(line, index_table, self.max_length))
            train_set = train_set.shuffle(buffer_size=50000)
            train_set = train_set.batch(self.batch_size)
            train_set = train_set.repeat(self.num_epochs)

            train_iterator = train_set.make_initializable_iterator("train_iterator")
            return train_iterator

        elif mode==tf.contrib.learn.ModeKeys.EVAL:
            val_set = tf.data.TextLineDataset(self.val_fnames)
            val_set = val_set.map(lambda line: self.parse_single_line(line, index_table, self.max_length))
            val_set = val_set.shuffle(buffer_size=10000)
            val_set = val_set.batch(self.batch_size)

            val_iterator = val_set.make_initializable_iterator("val_iterator")
            return val_iterator

        else:
            raise UserWarning

    def parse_single_line(self, line, index_table, max_length):
        """get single line, returns after padding and indexing"""
        splited = tf.string_split([line], delimiter="\t")
        query = tf.concat([["<SOS>", tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]]], axis=0)
        reply = tf.concat([["<SOS>", tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]]], axis=0)

        paddings = tf.constant([[0, 0], [0, max_length]])
        padded_query = tf.slice(tf.pad([query], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
        padded_reply = tf.slice(tf.pad([reply], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])

        indexed_query = tf.squeeze(index_table.lookup(padded_query))
        indexed_reply = tf.squeeze(index_table.lookup(padded_reply))

        return indexed_query, indexed_reply, tf.shape(indexed_query)[0], tf.shape(indexed_reply)[0]

    def parse_single_line_v2(self, line, index_table):
        splited = tf.string_split([line], delimiter="\t")
        query = tf.concat([["<SOS>", tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]]], axis=0)
        reply = tf.concat([["<SOS>", tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]]], axis=0)

        query = tf.squeeze(index_table.lookup(query))
        reply = tf.squeeze(index_table.lookup(reply))

        return {"input_queries": query,
                "input_replies": reply,
                "query_lengths": tf.shape(query)[0],
                "reply_lengths": tf.shape(reply)[0]}

    def get_fnames(self, dir):
        if os.path.isdir(dir):
            return [os.path.join(dir, fname) for fname in sorted(os.listdir(dir))]
        elif os.path.isfile(dir):
            return [dir]
        else:
            raise FileNotFoundError
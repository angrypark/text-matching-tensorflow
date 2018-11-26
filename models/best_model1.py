import os
import numpy as np
import tensorlow as tf

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask


"""
Best Model 1.
:author: @angrypark
:architecture: Dual Encoder Bi-directional GRU + Dense layer
:rnn: 512 dim * 2(bi-directional) * 2(dual encoder)
:dense_input: 2048(rnn last state) + 1(forward matmul) + 1(backward matmul) = 2050 dim
:dense_output: 1024 dim
:dense_activation_type: relu
"""


class BestModel1(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(BestModel1, self).__init__(dataset, config)
        self.mode = mode
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Build index table
        index_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self.config.vocab_list,
            num_oov_buckets=0,
            default_value=0
        )

        # Data generator
        self.data_iterator = self.data.get_data_iterator(index_table, mode=self.mode)

        # Inputs
        with tf.variable_scope("inputs"):
            next_batch = self.data_iterator.get_next()

            # Size: [batch_size, max_length]
            self.input_queries = tf.placeholder_with_default(
                next_batch["input_queries"],
                [None, self.config.max_length],
                name="input_queries")

            # Size: [batch_size, max_length]
            self.input_replies = tf.placeholder_with_default(
                next_batch["input_replies"],
                [None, self.config.max_length],
                name="input_replies")

            # Size: [batch_size]
            self.query_lengths = tf.placeholder_with_default(
                tf.squeeze(next_batch["query_lengths"]),
                [None],
                name="query_lengths")

            # Size: [batch_size]
            self.reply_lengths = tf.placeholder_with_default(
                tf.squeeze(next_batch["reply_lengths"]),
                [None],
                name="reply_lengths")

        with tf.variable_scope("properties"):
            # Current batch length
            cur_batch_length = tf.shape(self.input_queries)[0]

            # Learning rate and optimizer
            self.learning_rate = tf.train.exponential_decay(
                self.config.learning_rate,
                self.global_step_tensor,
                decay_steps=100000,
                decay_rate=0.96)

            




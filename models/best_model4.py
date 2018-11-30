# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from tensor2tensor.models.lstm import lstm_bid_encoder

from models.base import Model
from models.model_helper import (
    get_embeddings, make_negative_mask, gelu, dropout_lstm_cell
)


"""
Best Model 4.
:author: @shuuki4
:architecture: Dual Encoder 2 layer Bi-directional GRU + Dense layer + l2 norm
:rnn: 256(GRU) * 2(bi-directional) * 2(two layer) = 1024
:dense_input: 2048(rnn last state) + 1024(q-r) + 1024(q*r) + 1(q·r) = 4097
:dense_output: 1024 dim
:dense_activation_type: relu
"""


class BestModel4(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(BestModel4, self).__init__(dataset, config)
        if mode == "train":
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        elif (mode == "val") | (mode == tf.contrib.learn.ModeKeys.EVAL):
            self.mode = tf.contrib.learn.ModeKeys.EVAL
        else:
            raise NotImplementedError()
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Index table
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

            # Dropout rate
            self.embed_dropout_keep_prob = tf.placeholder(
                tf.float64, name="embed_dropout_keep_prob")
            self.lstm_dropout_keep_prob = tf.placeholder(
                tf.float32, name="lstm_dropout_keep_prob")
            self.dense_dropout_keep_prob = tf.placeholder(
                tf.float32, name="dense_dropout_keep_prob")
            self.num_negative_samples = tf.placeholder(
                tf.int32, name="num_negative_samples")

        # Properties
        with tf.variable_scope("properties"):
            # Current batch length
            cur_batch_length = tf.shape(self.input_queries)[0]

            # Learning rate and optimizer
            self.learning_rate = tf.train.exponential_decay(
                self.config.learning_rate,
                self.global_step_tensor,
                decay_steps=100000,
                decay_rate=0.96)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Embedding layer
        with tf.variable_scope("embedding_layer"):
            embeddings = tf.Variable(get_embeddings(self.config.vocab_list,
                                                    self.config.pretrained_embed_dir,
                                                    self.config.vocab_size,
                                                    self.config.embed_dim),
                                     trainable=True, name="embeddings")
            embeddings = tf.nn.dropout(embeddings,
                                       keep_prob=self.embed_dropout_keep_prob,
                                       noise_shape=[self.config.vocab_size, 1])
            self.queries_embedded = tf.to_float(tf.nn.embedding_lookup(embeddings,
                                       self.input_queries,
                                       name="queries_embedded"))
            self.replies_embedded = tf.to_float(tf.nn.embedding_lookup(embeddings,
                                       self.input_replies,
                                       name="replies_embedded"))

        # Query 2 layer bi-directional gru layer
        with tf.variable_scope("query_gru_layer"):
            query_cell_fw = tf.contrib.rnn.MultiRNNCell(
                [dropout_lstm_cell(self.config.lstm_dim / 2,
                                   self.lstm_dropout_keep_prob,
                                   cell_type="gru")
                 for _ in range(2)]
            )

            query_cell_bw = tf.contrib.rnn.MultiRNNCell(
                [dropout_lstm_cell(self.config.lstm_dim / 2,
                                   self.lstm_dropout_keep_prob,
                                   cell_type="gru")
                 for _ in range(2)]
            )

            _, (query_fw_state, query_bw_state) = tf.nn.bidirectional_dynamic_rnn(
                query_cell_fw,
                query_cell_bw,
                self.queries_embedded,
                self.query_lengths,
                dtype=tf.float32,
                time_major=False)

            query_fw_state_concat = tf.concat(query_fw_state, 1)
            query_bw_state_concat = tf.concat(query_bw_state, 1)

            # self.queries_encoded = tf.cast(
            #     tf.concat([query_fw_state_concat, query_bw_state_concat],
            #               1), tf.float64)
            self.queries_encoded = tf.concat([query_fw_state_concat, query_bw_state_concat], 1)
            

        # Reply 2 layer bi-directional gru layer
        with tf.variable_scope("reply_gru_layer"):
            reply_cell_fw = tf.contrib.rnn.MultiRNNCell(
                [dropout_lstm_cell(self.config.lstm_dim / 2,
                                   self.lstm_dropout_keep_prob,
                                   cell_type="gru")
                 for _ in range(2)]
            )

            reply_cell_bw = tf.contrib.rnn.MultiRNNCell(
                [dropout_lstm_cell(self.config.lstm_dim / 2,
                                   self.lstm_dropout_keep_prob,
                                   cell_type="gru")
                 for _ in range(2)]
            )

            _, (reply_fw_state, reply_bw_state) = tf.nn.bidirectional_dynamic_rnn(
                reply_cell_fw,
                reply_cell_bw,
                self.replies_embedded,
                self.reply_lengths,
                dtype=tf.float32,
                time_major=False)

            reply_fw_state_concat = tf.concat(reply_fw_state, 1)
            reply_bw_state_concat = tf.concat(reply_bw_state, 1)

            # self.replies_encoded = tf.cast(
            #     tf.concat([reply_fw_state_concat, reply_bw_state_concat],
            #               1), tf.float64)
            self.replies_encoded = tf.concat([reply_fw_state_concat, reply_bw_state_concat], 1)

        # Negative sampling
        with tf.variable_scope("sampling"):
            positive_mask = tf.eye(cur_batch_length)
            negative_mask = make_negative_mask(
                tf.zeros([cur_batch_length, cur_batch_length]),
                method=self.config.negative_sampling,
                num_negative_samples=self.num_negative_samples
            )
            negative_queries_indices, negative_replies_indices = tf.split(
                tf.where(tf.not_equal(negative_mask, 0)), [1, 1], 1)

            self.distances = tf.matmul(self.queries_encoded, self.replies_encoded,
                                       transpose_b=True)
            self.distances_flat = tf.reshape(self.distances, [-1])

            self.positive_distances = tf.gather(self.distances_flat, tf.where(
                tf.reshape(positive_mask, [-1])))

            self.negative_distances = tf.gather(self.distances_flat, tf.where(
                tf.reshape(negative_mask, [-1])))

            self.negative_queries_indices = tf.squeeze(negative_queries_indices)
            self.negative_replies_indices = tf.squeeze(negative_replies_indices)

        # Dense inputs
        with tf.variable_scope("dense_inputs"):
            self.positive_inputs = tf.concat([
                self.queries_encoded,
                self.positive_distances,
                self.queries_encoded - self.replies_encoded,
                tf.multiply(self.queries_encoded, self.replies_encoded),
                self.replies_encoded], 1)

            self.negative_queries_encoded = tf.reshape(tf.nn.embedding_lookup(
                self.queries_encoded, self.negative_queries_indices),
                [tf.shape(negative_queries_indices)[0], self.config.lstm_dim * 2])

            self.negative_replies_encoded = tf.reshape(tf.nn.embedding_lookup(
                self.replies_encoded, self.negative_replies_indices),
                [tf.shape(negative_queries_indices)[0], self.config.lstm_dim * 2])

            self.negative_inputs = tf.concat([
                self.negative_queries_encoded,
                self.negative_distances,
                self.negative_queries_encoded - self.negative_replies_encoded,
                tf.multiply(
                    self.negative_queries_encoded, self.negative_replies_encoded),
                self.negative_replies_encoded], 1)

        with tf.variable_scope("prediction"):
            self.hidden_outputs = tf.layers.dense(
                tf.concat([self.positive_inputs, self.negative_inputs], 0),
                1024,
                tf.nn.relu,
                name="dense_layer"
            )
            self.logits = tf.layers.dense(self.hidden_outputs,
                                          2,
                                          name="output_layer")
            labels = tf.concat(
                [tf.ones([tf.shape(self.positive_inputs)[0]], tf.int32),
                 tf.zeros([tf.shape(self.negative_inputs)[0]], tf.int32)], 0)

            self.labels = tf.one_hot(labels, 2, dtype=tf.int32)

            self.probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.probs, 1)

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                           logits=self.logits))
            l2_vars = [v for v in tf.trainable_variables()
                       if 'bias' not in v.name and 'embed' not in v.name]
            l2_loss = tf.add_n([tf.to_float(tf.nn.l2_loss(v)) for v in l2_vars])

            self.loss = tf.to_float(loss) + 1e-6 * l2_loss

            self.train_step = self.optimizer.minimize(
                self.loss, global_step=self.global_step_tensor)

        with tf.variable_scope("score"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                           name="accuracy")

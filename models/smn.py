import os
import numpy as np
import tensorflow as tf

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask

class SMN(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(SMN, self).__init__(dataset, config)
        self.mode = mode
        self.build_model()
        self.init_saver()

    def build_model(self):
        # build index table
        index_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self.config.vocab_list,
            num_oov_buckets=0,
            default_value=0)

        # get data iterator
        self.data_iterator = self.data.get_data_iterator(index_table, mode=self.mode)

        with tf.variable_scope("inputs"):
            # get next batch if there is no feeded data
            next_batch = self.data_iterator.get_next()
            self.input_queries = tf.placeholder_with_default(next_batch["input_queries"],
                                                             [None, self.config.max_length],
                                                             name="input_queries")
            self.input_replies = tf.placeholder_with_default(next_batch["input_replies"],
                                                             [None, self.config.max_length],
                                                             name="input_replies")
            self.query_lengths = tf.placeholder_with_default(tf.squeeze(next_batch["query_lengths"]),
                                                             [None],
                                                             name="query_lengths")
            self.reply_lengths = tf.placeholder_with_default(tf.squeeze(next_batch["reply_lengths"]),
                                                             [None],
                                                             name="reply_lengths")

            # get hyperparams
            self.embed_dropout_keep_prob = tf.placeholder(tf.float64, name="embed_dropout_keep_prob")
            self.lstm_dropout_keep_prob = tf.placeholder(tf.float32, name="lstm_dropout_keep_prob")
            self.dense_dropout_keep_prob = tf.placeholder(tf.float32, name="dense_dropout_keep_prob")
            self.num_negative_samples = tf.placeholder(tf.int32, name="num_negative_samples")

        with tf.variable_scope("properties"):
            # length properties
            cur_batch_length = tf.shape(self.input_queries)[0]

            # learning rate and optimizer
            learning_rate =  tf.train.exponential_decay(self.config.learning_rate,
                                                        self.global_step_tensor,
                                                        decay_steps=100000, decay_rate=0.96)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # embedding layer
        with tf.variable_scope("embedding"):
            embeddings = tf.Variable(get_embeddings(self.config.vocab_list,
                                                    self.config.pretrained_embed_dir,
                                                    self.config.vocab_size,
                                                    self.config.embed_dim),
                                     trainable=True,
                                     name="embeddings")
            embeddings = tf.nn.dropout(embeddings,
                                       keep_prob=self.embed_dropout_keep_prob,
                                       noise_shape=[tf.shape(embeddings)[0], 1])
            queries_embedded = tf.to_float(tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded"))
            replies_embedded = tf.to_float(tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded"))

        # gru layer
        with tf.variable_scope("gru_layer"):
            sentence_gru_cell = tf.nn.rnn_cell.GRUCell(self.config.lstm_dim, kernel_initializer=tf.initializers.orthogonal(), reuse=tf.AUTO_REUSE)
            sentence_gru_cell = tf.contrib.rnn.DropoutWrapper(sentence_gru_cell, 
                                                              input_keep_prob=self.lstm_dropout_keep_prob)
            self.query_rnn_outputs, _ = tf.nn.dynamic_rnn(sentence_gru_cell,
                                                          queries_embedded,
                                                          sequence_length=self.query_lengths,
                                                          dtype=tf.float32,
                                                          scope="sentence_gru")
            self.reply_rnn_outputs, _ = tf.nn.dynamic_rnn(sentence_gru_cell,
                                                          replies_embedded,
                                                          sequence_length=self.reply_lengths,
                                                          dtype=tf.float32,
                                                          scope="sentence_gru")

        # negative sampling
        with tf.variable_scope("negative_sampling"):
            negative_mask = make_negative_mask(tf.zeros([cur_batch_length, cur_batch_length]),
                                               method=self.config.negative_sampling,
                                               num_negative_samples=self.num_negative_samples)
            negative_queries_indices, negative_replies_indices = tf.split(tf.where(tf.not_equal(negative_mask, 0)),
                                                                          [1, 1], 1)

            self.negative_queries_indices = tf.squeeze(negative_queries_indices)
            self.negative_replies_indices = tf.squeeze(negative_replies_indices)
            self.num_negatives = tf.shape(self.negative_replies_indices)[0]

            queries_embedded_neg = tf.nn.embedding_lookup(queries_embedded, self.negative_queries_indices)
            replies_embedded_neg = tf.nn.embedding_lookup(replies_embedded, self.negative_replies_indices)

            self.query_rnn_outputs_neg = tf.reshape(tf.nn.embedding_lookup(self.query_rnn_outputs, self.negative_queries_indices), 
                                                    [self.num_negatives, self.config.max_length, self.config.lstm_dim])
            self.reply_rnn_outputs_neg = tf.reshape(tf.nn.embedding_lookup(self.reply_rnn_outputs, self.negative_replies_indices),
                                                    [self.num_negatives, self.config.max_length, self.config.lstm_dim])

        # build matrix for convolution
        with tf.variable_scope("matrix"):
            A_matrix = tf.get_variable("A_matrix_v",
                                       shape=(self.config.lstm_dim, self.config.lstm_dim),
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       dtype=tf.float32)
            
            replies_embedded_transposed = tf.transpose(replies_embedded, [0, 2, 1])
            reply_rnn_outputs_transposed = tf.transpose(self.reply_rnn_outputs, [0, 2, 1])
            replies_embedded_neg_transposed = tf.transpose(replies_embedded_neg, [0, 2, 1])
            reply_rnn_outputs_neg_transposed = tf.transpose(self.reply_rnn_outputs_neg, [0, 2, 1])
            
            embed_matrix = tf.matmul(queries_embedded,
                                     replies_embedded_transposed)
            
            rnn_outputs = tf.einsum("aij,jk->aik", self.query_rnn_outputs, A_matrix)
            rnn_outputs = tf.matmul(rnn_outputs, reply_rnn_outputs_transposed)
            self.matrix_stacked = tf.stack([embed_matrix, rnn_outputs], axis=3, name="matrix_stacked")

        # build negative matrix for convolution
        with tf.variable_scope("matrix", reuse=True):
            A_matrix_neg = tf.get_variable("A_matrix_v",
                                           shape=(self.config.lstm_dim, self.config.lstm_dim),
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           dtype=tf.float32)
            
            embed_matrix_neg = tf.matmul(queries_embedded_neg,
                                         replies_embedded_neg_transposed)
            
            rnn_outputs_neg = tf.einsum("aij,jk->aik", self.query_rnn_outputs_neg, A_matrix_neg)
            rnn_outputs_neg = tf.matmul(rnn_outputs_neg, reply_rnn_outputs_neg_transposed)
            self.matrix_stacked_neg = tf.stack([embed_matrix_neg, rnn_outputs_neg], axis=3, name="matrix_stacked_neg")

        # cnn layer
        with tf.variable_scope("convolution_layer"):
            conv = tf.layers.conv2d(self.matrix_stacked,
                                    filters=8,
                                    kernel_size=(3, 3),
                                    padding="VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    reuse=None,
                                    name="conv")
            pooled = tf.layers.max_pooling2d(conv, (3, 3), strides=(3, 3), padding="VALID", name="max_pooling")
            self.hidden_outputs = tf.expand_dims(tf.layers.dense(tf.contrib.layers.flatten(pooled),
                                             50,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer()), -1)

        # cnn layer
        with tf.variable_scope("convolution_layer", reuse=True):
            conv_neg = tf.layers.conv2d(self.matrix_stacked_neg,
                                    filters=8,
                                    kernel_size=(3, 3),
                                    padding="VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    reuse=True,
                                    name="conv")
            pooled_neg = tf.layers.max_pooling2d(conv_neg, (3, 3), strides=(3, 3), padding="VALID", name="max_pooling_neg")
            self.hidden_outputs_neg = tf.expand_dims(tf.layers.dense(tf.contrib.layers.flatten(pooled_neg),
                                                  50,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  reuse=True), -1)

        # matching gru layer
        with tf.variable_scope("matching_gru_layer"):
            matching_gru_cell = tf.nn.rnn_cell.GRUCell(self.config.lstm_dim,
                                                       kernel_initializer=tf.initializers.orthogonal(),
                                                       name="gru_cell",
                                                       reuse=tf.AUTO_REUSE)

            _, positive_state = tf.nn.dynamic_rnn(matching_gru_cell,
                                                  self.hidden_outputs,
                                                  dtype=tf.float32,
                                                  scope="matching_gru")

            _, negative_state = tf.nn.dynamic_rnn(matching_gru_cell,
                                                  self.hidden_outputs_neg,
                                                  dtype=tf.float32,
                                                  scope="matching_gru")

            self.positive_logits = tf.layers.dense(positive_state,
                                                   2,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   name="predict")

            self.negative_logits = tf.layers.dense(negative_state,
                                                   2,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   name="predict",
                                                   reuse=True)

        # build loss
        with tf.variable_scope("loss"):
            self.logits = tf.concat([self.positive_logits, self.negative_logits], 0)
            self.positive_probs = tf.nn.softmax(self.positive_logits)
            self.probs = tf.nn.softmax(self.logits)
            labels = tf.concat([tf.ones([tf.shape(self.positive_logits)[0]], tf.float64),
                                tf.zeros([tf.shape(self.negative_logits)[0]], tf.float64)], 0)
            self.labels = tf.one_hot(tf.to_int32(labels), 2)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(losses)
            self.train_step = self.optimizer.minimize(self.loss)
            
        with tf.variable_scope("score"):
            self.predictions = tf.argmax(self.probs, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
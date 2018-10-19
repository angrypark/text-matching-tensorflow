import os
import numpy as np
import tensorflow as tf

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask


class DualEncoderLSTM(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(DualEncoderLSTM, self).__init__(dataset, config)
        if mode == "train":
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        elif (mode == "val") | (mode == tf.contrib.learn.ModeKeys.EVAL):
            self.mode = tf.contrib.learn.ModeKeys.EVAL
        else:
            raise NotImplementedError()
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

        # get inputs
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
            self.embed_dropout_keep_prob = tf.placeholder(tf.float32, name="embed_dropout_keep_prob")
            self.lstm_dropout_keep_prob = tf.placeholder(tf.float32, name="lstm_dropout_keep_prob")
            self.num_negative_samples = tf.placeholder(tf.int32, name="num_negative_samples")
            self.add_echo = tf.placeholder(tf.bool, name="add_echo")

        with tf.variable_scope("properties"):
            # length properties
            cur_batch_length = tf.shape(self.input_queries)[0]
            query_max_length = tf.shape(self.input_queries)[1]
            reply_max_length = tf.shape(self.input_replies)[1]

            # learning rate and optimizer
            learning_rate =  tf.train.exponential_decay(self.config.learning_rate,
                                                        self.global_step_tensor,
                                                        decay_steps=100000, decay_rate=0.9)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # embedding layer
        with tf.variable_scope("embedding"):
            embeddings = tf.Variable(get_embeddings(self.config.vocab_list,
                                                    self.config.pretrained_embed_dir,
                                                    self.config.vocab_size,
                                                    self.config.embed_dim),
                                     trainable=True,
                                     name="embeddings")
            queries_embedded = tf.to_float(tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded"))
            replies_embedded = tf.to_float(tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded"))

        # build LSTM layer
        with tf.variable_scope("lstm_layer") as vs:
            query_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                      forget_bias=2.0,
                                                      use_peepholes=True,
                                                      state_is_tuple=True,
                                                      # initializer=tf.orthogonal_initializer(),
                                                     )
            query_lstm_cell = tf.contrib.rnn.DropoutWrapper(query_lstm_cell,
                                                           input_keep_prob=self.lstm_dropout_keep_prob)
            reply_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                      forget_bias=2.0,
                                                      use_peepholes=True,
                                                      state_is_tuple=True,
                                                      # initializer=tf.orthogonal_initializer(),
                                                      reuse=True)
            reply_lstm_cell = tf.contrib.rnn.DropoutWrapper(reply_lstm_cell,
                                                           input_keep_prob=self.lstm_dropout_keep_prob)
            _, queries_encoded = tf.nn.dynamic_rnn(
                cell=query_lstm_cell,
                inputs=queries_embedded,
                sequence_length=tf.cast(self.query_lengths, tf.float32),
                dtype=tf.float32,
            )
            _, replies_encoded = tf.nn.dynamic_rnn(
                cell=reply_lstm_cell,
                inputs=replies_embedded,
                sequence_length=tf.cast(self.reply_lengths, tf.float32),
                dtype=tf.float32,
            )

            _, echo_encoded = tf.nn.dynamic_rnn(
                cell=reply_lstm_cell,
                inputs=queries_embedded,
                sequence_length=tf.cast(tf.squeeze(self.query_lengths), tf.float32),
                dtype=tf.float32)

            self.queries_encoded = tf.cast(queries_encoded.h, tf.float64)
            self.replies_encoded = tf.cast(replies_encoded.h, tf.float64)
            self.echo_encoded = tf.cast(echo_encoded.h, tf.float64)

        # build dense layer
        with tf.variable_scope("dense_layer"):
            M = tf.get_variable("M",
                                shape=[self.config.lstm_dim, self.config.lstm_dim],
                                initializer=tf.initializers.truncated_normal())
            self.queries_encoded = tf.matmul(self.queries_encoded, tf.cast(M, tf.float64))

        with tf.variable_scope("sampling"):
            self.distances = tf.matmul(self.queries_encoded, self.replies_encoded, transpose_b=True)
            self.echo_distances = tf.matmul(self.queries_encoded, self.echo_encoded, transpose_b=True)
            positive_mask = tf.reshape(tf.eye(cur_batch_length), [-1])
            negative_mask = tf.reshape(make_negative_mask(self.distances,
                                                          method=self.config.negative_sampling,
                                                          num_negative_samples=self.num_negative_samples), [-1])

        with tf.variable_scope("prediction"):
            distances_flattened = tf.reshape(self.distances, [-1])
            echo_distances_flattened = tf.reshape(self.echo_distances, [-1])
            self.positive_logits = tf.gather(distances_flattened, tf.where(positive_mask), 1)
            self.negative_logits = tf.gather(distances_flattened, tf.where(negative_mask), 1)
            self.echo_logits = tf.gather(echo_distances_flattened, tf.where(positive_mask), 1)

            self.logits = tf.cond(self.add_echo, 
                                  lambda: tf.concat([self.positive_logits,
                                                     self.negative_logits,
                                                     self.echo_logits], axis=0),
                                  lambda: tf.concat([self.positive_logits,
                                                     self.negative_logits], axis=0))
            self.labels = tf.cond(self.add_echo,
                                  lambda: tf.concat([tf.ones_like(self.positive_logits),
                                                     tf.zeros_like(self.negative_logits),
                                                     tf.zeros_like(self.echo_logits)], axis=0),
                                  lambda: tf.concat([tf.ones_like(self.positive_logits),
                                                     tf.zeros_like(self.negative_logits)], axis=0))
            
            self.positive_probs = tf.sigmoid(self.positive_logits)
            self.echo_probs = tf.sigmoid(self.echo_logits)

            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.cast(self.probs > 0.5, dtype=tf.int32)
            
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            # gvs = self.optimizer.compute_gradients(self.loss)
            # capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
            # self.train_step = self.optimizer.apply_gradients(capped_gvs)
            self.train_step = self.optimizer.minimize(self.loss)
            
        with tf.variable_scope("score"):
            correct_predictions = tf.equal(self.predictions, tf.to_int32(self.labels))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

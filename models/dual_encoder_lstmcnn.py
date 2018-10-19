import os
import numpy as np
import tensorflow as tf

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask


class DualEncoderLSTMCNN(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(DualEncoderLSTMCNN, self).__init__(dataset, config)
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
            self.embed_dropout_keep_prob = tf.placeholder(tf.float64, name="embed_dropout_keep_prob")
            self.lstm_dropout_keep_prob = tf.placeholder(tf.float32, name="lstm_dropout_keep_prob")
            self.dense_dropout_keep_prob = tf.placeholder(tf.float32, name="dense_dropout_keep_prob")
            self.num_negative_samples = tf.placeholder(tf.int32, name="num_negative_samples")

        with tf.variable_scope("properties"):
            # length properties
            cur_batch_length = tf.shape(self.input_queries)[0]
            query_max_length = tf.shape(self.input_queries)[1]
            reply_max_length = tf.shape(self.input_replies)[1]

            # learning rate and optimizer
            learning_rate = tf.train.exponential_decay(self.config.learning_rate,
                                                       self.global_step_tensor,
                                                       decay_steps=20000, decay_rate=0.96)
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
                                       noise_shape=[90000, 1])
            queries_embedded = tf.to_float(
                tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded"))
            replies_embedded = tf.to_float(
                tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded"))
            self.queries_embedded = queries_embedded
            self.replies_embedded = replies_embedded

        # build LSTM layer
        with tf.variable_scope("lstm_layer") as vs:
            query_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                      forget_bias=2.0,
                                                      use_peepholes=True,
                                                      state_is_tuple=True)
            query_lstm_cell = tf.contrib.rnn.DropoutWrapper(query_lstm_cell,
                                                            input_keep_prob=self.lstm_dropout_keep_prob)
            reply_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                      forget_bias=2.0,
                                                      use_peepholes=True,
                                                      state_is_tuple=True,
                                                      reuse=True)
            reply_lstm_cell = tf.contrib.rnn.DropoutWrapper(reply_lstm_cell,
                                                            input_keep_prob=self.lstm_dropout_keep_prob)
            queries_encoded, queries_state = tf.nn.dynamic_rnn(
                cell=query_lstm_cell,
                inputs=queries_embedded,
                sequence_length=tf.cast(self.query_lengths, tf.float32),
                dtype=tf.float32,
            )
            replies_encoded, replies_state = tf.nn.dynamic_rnn(
                cell=reply_lstm_cell,
                inputs=replies_embedded,
                sequence_length=tf.cast(self.reply_lengths, tf.float32),
                dtype=tf.float32,
            )

            self.queries_encoded = tf.expand_dims(queries_encoded, -1)
            self.replies_encoded = tf.expand_dims(replies_encoded, -1)

        # Create a convolution + maxpool layer for each filter size
        queries_pooled_outputs = list()
        replies_pooled_outputs = list()

        for i, filter_size in enumerate([1, 2, 3, 4, 5]):
            filter_shape = [filter_size, self.config.lstm_dim, 1, 128]

            # queries
            with tf.name_scope("conv-maxpool-query-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[128]), name="bias")
                conv = tf.nn.conv2d(
                    self.queries_encoded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                queries_pooled_outputs.append(pooled)

            # replies
            with tf.name_scope("conv-maxpool-reply-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[128]), name="bias")
                conv = tf.nn.conv2d(
                    self.replies_encoded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                replies_pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = 128 * 5

        self.queries_conv_output = tf.reshape(tf.concat(queries_pooled_outputs, 3), [-1, num_filters_total])
        self.replies_conv_output = tf.reshape(tf.concat(replies_pooled_outputs, 3), [-1, num_filters_total])

        with tf.variable_scope("sampling"):
            positive_mask = tf.reshape(tf.eye(cur_batch_length), [-1])
            negative_mask = make_negative_mask(tf.zeros([cur_batch_length, cur_batch_length]),
                                               method=self.config.negative_sampling,
                                               num_negative_samples=self.num_negative_samples)
            negative_queries_indices, negative_replies_indices = tf.split(tf.where(tf.not_equal(negative_mask, 0)),
                                                                          [1, 1], 1)

            self.negative_queries_indices = tf.squeeze(negative_queries_indices)
            self.negative_replies_indices = tf.squeeze(negative_replies_indices)

            self.distances = tf.matmul(queries_state.h, replies_state.h, transpose_b=True)
            self.distances_flattened = tf.reshape(self.distances, [-1])
            self.positive_distances = tf.gather(self.distances_flattened, tf.where(positive_mask), 1)
            self.negative_distances = tf.gather(self.distances_flattened, tf.where(tf.reshape(negative_mask, [-1])), 1)

            self.positive_inputs = tf.concat([self.queries_conv_output,
                                              self.positive_distances,
                                              self.replies_conv_output], 1)
            self.negative_inputs = tf.reshape(
                tf.concat([tf.nn.embedding_lookup(self.queries_conv_output, self.negative_queries_indices),
                           self.negative_distances,
                           tf.nn.embedding_lookup(self.replies_conv_output, self.negative_replies_indices)], 1),
                [tf.shape(negative_queries_indices)[0], num_filters_total * 2 + 1])

            self.num_positives = tf.shape(self.positive_inputs)[0]
            self.num_negatives = tf.shape(self.negative_inputs)[0]

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[2 * num_filters_total + 1, 100],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="bias")
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(tf.concat([self.positive_inputs,
                                                                       self.negative_inputs], 0),
                                                            W,
                                                            b,
                                                            name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dense_dropout_keep_prob, name="hidden_output_drop")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[100, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")

            self.positive_logits, self.negative_logits = tf.split(self.logits, [self.num_positives, self.num_negatives])
            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.to_int32(self.probs > 0.5, name="predictions")

            labels = tf.concat([tf.ones([self.num_positives], tf.float64),
                                tf.zeros([self.num_negatives], tf.float64)], 0)

            self.labels = tf.to_int32(labels)

        with tf.variable_scope("loss"):
            self.positive_scores = tf.expand_dims(self.positive_logits, 1)
            self.negative_scores = self.negative_logits
            self.ranking_loss = tf.reduce_sum(tf.maximum(0.0, 
                                                         self.config.hinge_loss - self.positive_scores + self.negative_scores))
            l2_vars = [v for v in tf.trainable_variables()
                       if 'bias' not in v.name and 'embedding' not in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in l2_vars])
            
            self.loss = self.ranking_loss + l2_loss
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        with tf.variable_scope("score"):
            correct_predictions = tf.equal(self.predictions, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

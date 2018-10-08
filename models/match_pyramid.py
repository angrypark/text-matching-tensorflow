import os
import numpy as np
import tensorflow as tf

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask


class MatchPyramid(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(MatchPyramid, self).__init__(dataset, config)
        if mode == "train":
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        elif (mode == "val") | (mode == tf.contrib.learn.ModeKeys.EVAL):
            self.mode = tf.contrib.learn.ModeKeys.EVAL
        else:
            raise NotImplementedError()
        self.filter_sizes = [int(filter_size) for filter_size in self.config.filter_sizes.split(",")]
        self.build_model()
        self.init_saver()
    
    def dynamic_pooling_index(self, query_lengths, reply_lengths, max_length):
        query_stride = 1.0 * max_length / query_lengths
        reply_stride = 1.0 * max_length / reply_lengths
        
            

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
            queries_embedded = tf.expand_dims(tf.to_float(tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded")), -1)
            replies_embedded = tf.expand_dims(tf.to_float(tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded")), -1)

        # build CNN layer
        with tf.variable_scope("convolution_layer"):
            queries_pooled_outputs = list()
            replies_pooled_outputs = list()
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.config.embed_dim, 1, self.config.num_filters]
                with tf.name_scope("conv-maxpool-query-{}".format(filter_size)):
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = tf.nn.conv2d(queries_embedded, 
                                        W, 
                                        strides=[1, 1, 1, 1], 
                                        padding="VALID", 
                                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(h, 
                                            ksize=[1, self.config.max_length - filter_size + 1, 1, 1], 
                                            strides=[1, 1, 1, 1], 
                                            padding="VALID", 
                                            name="pool")
                    queries_pooled_outputs.append(pooled)
                    
                with tf.name_scope("conv-maxpool-reply-{}".format(filter_size)):
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = tf.nn.conv2d(replies_embedded, 
                                        W, 
                                        strides=[1, 1, 1, 1], 
                                        padding="VALID", 
                                        name="conv", 
                                        # reuse=True,
                                       )
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(h, 
                                            ksize=[1, self.config.max_length - filter_size + 1, 1, 1], 
                                            strides=[1, 1, 1, 1], 
                                            padding="VALID", 
                                            name="pool")
                    replies_pooled_outputs.append(pooled)
                    
                    # conv_echo = tf.nn.conv2d(queries_embedded, 
                    #                     W, 
                    #                     strides=[1, 1, 1, 1], 
                    #                     padding="VALID", 
                    #                     name="conv", 
                    #                     reuse=True)
                    # h_echo = tf.nn.relu(tf.nn.bias_add(conv_echo, b), name="relu_echo")
                    # pooled_echo = tf.nn.max_pool(h_echo, 
                    #                         ksize=[1, self.config.max_length - filter_size + 1, 1, 1], 
                    #                         strides=[1, 1, 1, 1], 
                    #                         padding="VALID", 
                    #                         name="pool_echo")
                    # echo_pooled_outputs.append(pooled_echo)
        
        # combine all pooled outputs
        num_filters_total = self.config.num_filters * len(self.filter_sizes)
        self.queries_encoded = tf.reshape(tf.concat(queries_pooled_outputs, 3), 
                                          [-1, num_filters_total], name="queries_encoded")
        self.replies_encoded = tf.reshape(tf.concat(replies_pooled_outputs, 3), 
                                          [-1, num_filters_total], name="replies_encoded")
        
        with tf.variable_scope("dense_layer"):
            M = tf.get_variable("M", 
                                shape=[num_filters_total, num_filters_total], 
                                initializer=tf.contrib.layers.xavier_initializer())
            self.queries_transformed = tf.matmul(self.queries_encoded, M)

        with tf.variable_scope("sampling"):
            self.distances = tf.matmul(self.queries_transformed, self.replies_encoded, transpose_b=True)
            # self.echo_distances = tf.matmul(self.queries_transformed, self.echo_encoded, transpose_b=True)
            positive_mask = tf.reshape(tf.eye(cur_batch_length), [-1])
            negative_mask = tf.reshape(make_negative_mask(self.distances,
                                                          method=self.config.negative_sampling,
                                                          num_negative_samples=self.num_negative_samples), [-1])

        with tf.variable_scope("prediction"):
            distances_flattened = tf.reshape(self.distances, [-1])
            # echo_distances_flattened = tf.reshape(self.echo_distances, [-1])
            self.positive_logits = tf.gather(distances_flattened, tf.where(positive_mask), 1)
            self.negative_logits = tf.gather(distances_flattened, tf.where(negative_mask), 1)
            
            self.logits = tf.concat([self.positive_logits, self.negative_logits], axis=0)
            self.labels = tf.concat([tf.ones_like(self.positive_logits), tf.zeros_like(self.negative_logits)], axis=0)
            
            # self.echo_logits = tf.gather(echo_distances_flattened, tf.where(positive_mask), 1)

            # self.logits = tf.cond(self.add_echo, 
            #                       lambda: tf.concat([self.positive_logits,
            #                                          self.negative_logits,
            #                                          self.echo_logits], axis=0),
            #                       lambda: tf.concat([self.positive_logits,
            #                                          self.negative_logits], axis=0))
            # self.labels = tf.cond(self.add_echo,
            #                       lambda: tf.concat([tf.ones_like(self.positive_logits),
            #                                          tf.zeros_like(self.negative_logits),
            #                                          tf.zeros_like(self.echo_logits)], axis=0),
            #                       lambda: tf.concat([tf.ones_like(self.positive_logits),
            #                                          tf.zeros_like(self.negative_logits)], axis=0))
            
            self.positive_probs = tf.sigmoid(self.positive_logits)
            #  self.echo_probs = tf.sigmoid(self.echo_logits)

            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.cast(self.probs>0.5, dtype=tf.int32)
            
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            gvs = self.optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
            self.train_step = self.optimizer.apply_gradients(capped_gvs)
            # self.train_step = self.optimizer.minimize(self.loss)
            
        with tf.variable_scope("score"):
            correct_predictions = tf.equal(self.predictions, tf.to_int32(self.labels))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

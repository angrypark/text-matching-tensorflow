import os
import numpy as np
import tensorflow as tf
from tensor2tensor.models import transformer
from tensor2tensor.utils import optimize
from tensor2tensor.utils import learning_rate

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask

class DualEncoderTransformerDense2(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(DualEncoderTransformerDense2, self).__init__(dataset, config)
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
                
            # get hparms from tensor2tensor.models.transformer
            hparams = transformer.transformer_small()
            hparams.batch_size = self.config.batch_size
            # hparams.learning_rate_decay_steps = 10000
            # hparams.learning_rate_minimum = 3e-5
            
            # learning rate
            lr = learning_rate.learning_rate_schedule(hparams)
            self.learning_rate = lr

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
            
            self.queries_embedded = queries_embedded
            self.replies_embedded = replies_embedded
        
        # transformer layer
        with tf.variable_scope("transformer"):
            queries_expanded = tf.expand_dims(queries_embedded, axis=2, name="queries_expanded")
            replies_expanded = tf.expand_dims(replies_embedded, axis=2, name="replies_expanded")
            
            hparams = transformer.transformer_small()
            hparams.set_hparam("batch_size", self.config.batch_size)
            hparams.set_hparam("hidden_size", self.config.embed_dim)
            encoder = transformer.TransformerEncoder(hparams, mode=self.mode)
            
            self.queries_encoded = encoder({"inputs": queries_expanded, 
                                             "targets": queries_expanded})[0]
            self.replies_encoded = encoder({"inputs": replies_expanded,
                                             "targets": replies_expanded})[0]
            
            self.queries_pooled = tf.squeeze(tf.nn.max_pool(self.queries_encoded, 
                                                 ksize=[1, self.config.max_length, 1, 1], 
                                                 strides=[1, 1, 1, 1], 
                                                 padding='VALID', 
                                                 name="queries_pooled"))
            self.replies_pooled = tf.squeeze(tf.nn.max_pool(self.replies_encoded, 
                                                 ksize=[1, self.config.max_length, 1, 1], 
                                                 strides=[1, 1, 1, 1], 
                                                 padding='VALID', 
                                                 name="replies_pooled"))

        with tf.variable_scope("sampling"):
            positive_mask = tf.eye(cur_batch_length)
            negative_mask = make_negative_mask(tf.zeros([cur_batch_length, cur_batch_length]),
                                               method=self.config.negative_sampling,
                                               num_negative_samples=self.num_negative_samples)
            negative_queries_indices, negative_replies_indices = tf.split(tf.where(tf.not_equal(negative_mask, 0)), [1, 1], 1)
            
            self.distances = tf.matmul(self.queries_pooled, self.replies_pooled, transpose_b=True)
            self.distances_flattened = tf.reshape(self.distances, [-1])
            
            self.positive_distances = tf.gather(self.distances_flattened, tf.where(tf.reshape(positive_mask, [-1])))
            self.negative_distances = tf.gather(self.distances_flattened, tf.where(tf.reshape(negative_mask, [-1])))
            
            self.negative_queries_indices = tf.squeeze(negative_queries_indices)
            self.negative_replies_indices = tf.squeeze(negative_replies_indices)
    
            self.positive_inputs = tf.concat([self.queries_pooled, self.positive_distances, self.replies_pooled], 1)
            self.negative_inputs = tf.reshape(tf.concat([tf.nn.embedding_lookup(self.queries_pooled, self.negative_queries_indices),
                                                         self.negative_distances,
                                                         tf.nn.embedding_lookup(self.replies_pooled, self.negative_replies_indices)], 1),
                                              [tf.shape(negative_queries_indices)[0], self.config.embed_dim * 2 + 1])
                                     
        with tf.variable_scope("prediction"):
            self.hidden_outputs = tf.layers.dense(tf.concat([self.positive_inputs, self.negative_inputs], 0), 
                                                  256,
                                                  tf.nn.relu,
                                                  name="hidden_layer")
            self.logits = tf.layers.dense(self.hidden_outputs,
                                          2,
                                          tf.nn.relu, name="output_layer")
            labels = tf.concat([tf.ones([tf.shape(self.positive_inputs)[0]], tf.float64),
                                tf.zeros([tf.shape(self.negative_inputs)[0]], tf.float64)], 0)

            self.labels = tf.one_hot(tf.to_int32(labels), 2)
            
            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.argmax(self.probs, 1)
            
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits))
            self.train_step = optimize.optimize(self.loss, lr, hparams, use_tpu=False)
            
        with tf.variable_scope("score"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
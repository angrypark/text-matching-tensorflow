import os
import numpy as np
import tensorflow as tf
from tensor2tensor.models import transformer
from tensor2tensor.utils import optimize
from tensor2tensor.utils import learning_rate

from models.base import Model
from models.model_helper import get_embeddings, make_negative_mask

class DualEncoderTransformer(Model):
    def __init__(self, dataset, config, mode=tf.contrib.learn.ModeKeys.TRAIN):
        super(DualEncoderTransformer, self).__init__(dataset, config)
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
            
            # learning rate
            lr = learning_rate.learning_rate_schedule(hparams)

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
            
            self.queries_pooled = tf.nn.max_pool(self.queries_encoded, 
                                                 ksize=[1, self.config.max_length, 1, 1], 
                                                 strides=[1, 1, 1, 1], 
                                                 padding='VALID', 
                                                 name="queries_pooled")
            self.replies_pooled = tf.nn.max_pool(self.replies_encoded, 
                                                 ksize=[1, self.config.max_length, 1, 1], 
                                                 strides=[1, 1, 1, 1], 
                                                 padding='VALID', 
                                                 name="replies_pooled")
            
            self.queries_flattened = tf.reshape(self.queries_pooled, [cur_batch_length, -1])
            self.replies_flattened = tf.reshape(self.replies_pooled, [cur_batch_length, -1])

        # build dense layer
        with tf.variable_scope("dense_layer"):
            M = tf.get_variable("M",
                                shape=[self.config.embed_dim, self.config.embed_dim],
                                initializer=tf.initializers.truncated_normal())
            M = tf.nn.dropout(M, self.dense_dropout_keep_prob)
            self.queries_transformed = tf.matmul(self.queries_flattened, M)

        with tf.variable_scope("sampling"):
            self.distances = tf.matmul(self.queries_transformed, self.replies_flattened, transpose_b=True)
            positive_mask = tf.reshape(tf.eye(cur_batch_length), [-1])
            negative_mask = tf.reshape(make_negative_mask(self.distances,
                                                          method=self.config.negative_sampling,
                                                          num_negative_samples=self.num_negative_samples), [-1])
            
        with tf.variable_scope("prediction"):
            distances_flattened = tf.reshape(self.distances, [-1])
            self.positive_logits = tf.gather(distances_flattened, tf.where(positive_mask), 1)
            self.negative_logits = tf.gather(distances_flattened, tf.where(negative_mask), 1)

            self.logits = tf.concat([self.positive_logits,
                                     self.negative_logits], axis=0)
            self.labels = tf.concat([tf.ones_like(self.positive_logits),
                                     tf.zeros_like(self.negative_logits)], axis=0)
            
            self.positive_probs = tf.sigmoid(self.positive_logits)

            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.cast(self.probs > 0.5, dtype=tf.int32)
            
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.train_step = optimize.optimize(self.loss, lr, hparams, use_tpu=False)
            
        with tf.variable_scope("score"):
            correct_predictions = tf.equal(self.predictions, tf.to_int32(self.labels))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
import os
import tensorflow as tf
import numpy as np
from gensim.models import FastText

from utils.utils import JamoProcessor

def get_embeddings(vocab_list_dir,
                   pretrained_embed_dir,
                   vocab_size,
                   embed_dim):
    embedding = np.random.uniform(-1/16, 1/16, [vocab_size, embed_dim])
    if os.path.isfile(pretrained_embed_dir) & os.path.isfile(vocab_list_dir):
        with open(vocab_list_dir, "r") as f:
            vocab_list = [word.strip() for word in f if len(word)>0]
        processor = JamoProcessor()
        ft = FastText.load(pretrained_embed_dir)
        num_oov = 0
        for i, vocab in enumerate(vocab_list):
            try:
                embedding[i, :] = ft.wv[processor.word_to_jamo(vocab)]
            except:
                num_oov += 1
        print("Pre-trained embedding loaded. Number of OOV : {} / {}".format(num_oov, len(vocab_list)))
    else:
        print("No pre-trained embedding found, initialize with random distribution")
    return embedding


def make_negative_mask(distances, num_negative_samples, method="random"):
    cur_batch_length = tf.shape(distances)[0]
    if method == "random":
        topk = tf.contrib.framework.sort(
            tf.nn.top_k(tf.random_uniform([cur_batch_length, cur_batch_length]), k=num_negative_samples).indices,
            axis=1)
        rows = tf.transpose(tf.reshape(tf.tile(tf.range(cur_batch_length), [num_negative_samples]),
                                       [num_negative_samples, cur_batch_length]))
        indices = tf.to_int64(tf.reshape(tf.concat([tf.expand_dims(rows, -1), tf.expand_dims(topk, -1)], axis=2),
                                         [num_negative_samples * cur_batch_length, 2]))
        mask = tf.sparse_to_dense(sparse_indices=indices,
                                  output_shape=[tf.to_int64(cur_batch_length), tf.to_int64(cur_batch_length)],
                                  sparse_values=tf.ones([(num_negative_samples * cur_batch_length)], 1))

        # drop positive
        mask = tf.multiply(mask, (1 - tf.eye(cur_batch_length)))

    elif method == "hard":
        topk = tf.contrib.framework.sort(tf.nn.top_k(distances, k=num_negative_samples + 1).indices, axis=1)
        rows = tf.transpose(tf.reshape(tf.tile(tf.range(cur_batch_length), [num_negative_samples + 1]),
                                       [num_negative_samples + 1, cur_batch_length]))
        indices = tf.to_int64(tf.reshape(tf.concat([tf.expand_dims(rows, -1), tf.expand_dims(topk, -1)], axis=2),
                                         [(num_negative_samples + 1) * cur_batch_length, 2]))
        mask = tf.sparse_to_dense(sparse_indices=indices,
                                  output_shape=[tf.to_int64(cur_batch_length), tf.to_int64(cur_batch_length)],
                                  sparse_values=tf.ones([((num_negative_samples + 1) * cur_batch_length)], 1))
        # drop positive
        mask = tf.multiply(mask, (1 - tf.eye(cur_batch_length)))

    else:
        raise NotImplementedError
    # elif method == "weighted":
    #     weight = tf.map_fn(lambda x: get_distance_weight(x, batch_size), tf.to_float(distances))
    #     mask = weight
    #         mask = tf.to_int32(tf.contrib.framework.sort(tf.expand_dims(tf.multinomial(weight, num_negative_samples+1), -1), axis=1))
    #         weighted_samples_indices = tf.to_int32(tf.contrib.framework.sort(tf.expand_dims(tf.multinomial(weight, num_negative_samples+1), -1), axis=1))
    #         row_indices = tf.expand_dims(tf.transpose(tf.reshape(tf.tile(tf.range(0, batch_size, 1), [num_negative_samples+1]), [num_negative_samples+1, batch_size])), -1)
    #         mask_indices = tf.to_int64(tf.squeeze(tf.reshape(tf.concat([row_indices, weighted_samples_indices], 2), [(num_negative_samples+1)*batch_size,1,2])))
    #         mask_sparse = tf.SparseTensor(mask_indices, [1]*((num_negative_samples+1)*batch_size), [batch_size,batch_size])
    #         mask = tf.sparse_tensor_to_dense(mask_sparse)
    #         drop_positive = tf.to_int32(tf.subtract(tf.ones([batch_size, batch_size]), tf.eye(batch_size)))
    #         mask = tf.multiply(mask, drop_positive)
    return mask

def get_optimizer(global_step, 
                  optimize_method="adam", 
                  learning_rate=1e-3, 
                  warm_up_steps=0, 
                  decay_method=None, 
                  decay_steps=0, 
                  decay_rate=0, 
                  end_learning_rate=0):
    # define learning rate with decay method
    if not decay_method:
        pass
    elif decay_method=="exponential":
        learning_rate = tf.train.exponential_decay(learning_rate, 
                                                   global_step, 
                                                   decay_steps=decay_steps,
                                                   decay_rate=decay_rate)
    elif decay_method=="step":
        learning_rate = tf.train.polynomial_decay(learning_rate,
                                                  global_step=global_step,
                                                  decay_steps=decay_steps,
                                                  end_learning_rate=end_learning_rate)
    else:
        raise NotImplementedError()

    # define optimizer
    if optimize_method == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
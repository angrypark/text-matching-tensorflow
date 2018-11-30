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
        with open(vocab_list_dir, "r", encoding="utf-8") as f:
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

def dropout_lstm_cell(hidden_size, lstm_dropout_keep_prob, cell_type="lstm"):
    if cell_type == "lstm":
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(hidden_size),
            input_keep_prob=lstm_dropout_keep_prob)
    elif cell_type == "gru":
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(hidden_size),
            input_keep_prob=lstm_dropout_keep_prob)

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(tf.constant(2.0, tf.float64))))
  return input_tensor * cdf

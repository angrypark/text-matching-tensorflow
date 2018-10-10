import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time

from models.base import get_model
from utils.logger import setup_logger
from utils.config import save_config


class BaseTrainer:
    def __init__(self, sess, preprocessor, data, config, summary_writer):
        self.sess = sess
        self.preprocessor = preprocessor
        self.data = data
        self.config = config
        self.summary_writer = summary_writer
        self.logger = setup_logger()

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(models, sess)
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, model, sess):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self, model, sess):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError


class MatchingModelTrainer(BaseTrainer):
    def __init__(self, sess, preprocessor, data, config, summary_writer):
        super(MatchingModelTrainer, self).__init__(sess, preprocessor, data, config, summary_writer)
        # get size of data
        self.train_size = data.train_size
        self.val_size = data.val_size
        self.batch_size = config.batch_size

        # initialize global step, epoch
        self.num_steps_per_epoch = (self.train_size - 1) // self.batch_size + 1
        self.cur_epoch = 1
        self.global_step = 1

        # for summary and logger
        self.summary_dict = dict()
        self.train_summary = "Epoch : {:2d} | Step : {:8d} | Train loss : {:.4f} | Train accuracy : {:.4f} "
        self.val_summary = "| Val loss : {:.4f} | Val accuracy : {:.4f} "

        # checkpoint_dir
        self.checkpoint_dir = config.checkpoint_dir

        # train, val iterator
        self.train_iterator = None
        self.val_iterator = None
        
        # load pretrained model
        self.infer_preprocessor = None
        
        self.use_weak_supervision = config.weak_supervision
        
        if self.use_weak_supervision:
            self.infer_model, self.infer_sess = self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        from collections import namedtuple
        import sys
        sys.path.append("/home/angrypark/")
        
        from tmp.data_loader import DataGenerator as _Dataset
        from tmp.trainer import MatchingModelTrainer as _Trainer
        from tmp.preprocessor import DynamicPreprocessor as _Preprocessor
        from tmp.utils.dirs import create_dirs as _create_dirs
        from tmp.utils.logger import SummaryWriter as _SummaryWriter
        from tmp.utils.config import load_config, save_config
        from tmp.models.base import get_model as _get_model
        from tmp.utils.utils import JamoProcessor
        from tmp.text.tokenizers import SentencePieceTokenizer
        from tmp.models.dual_encoder_lstm import DualEncoderLSTM as Model
        
        Config = namedtuple("config", ["sent_piece_model"])
        config = Config("/media/scatter/scatterdisk/tokenizer/sent_piece.100K.model")
        processor = JamoProcessor()
        tokenizer = SentencePieceTokenizer(config)
        
        base_dir = "/media/scatter/scatterdisk/reply_matching_model/runs/delstm_1024_nsrandom4_lr1e-3/"
        config_dir = base_dir + "config.json"
        best_model_dir = base_dir + "best_loss/best_loss.ckpt"
        model_config = load_config(config_dir)
        model_config.add_echo = False
        preprocessor = _Preprocessor(model_config)
        preprocessor.build_preprocessor()

        infer_config = load_config(config_dir)
        setattr(infer_config, "tokenizer", "SentencePieceTokenizer")
        setattr(infer_config, "soynlp_scores", "/media/scatter/scatterdisk/tokenizer/soynlp_scores.sol.100M.txt")
        infer_preprocessor = _Preprocessor(infer_config)
        infer_preprocessor.build_preprocessor()
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with graph.as_default():
            data = _Dataset(preprocessor, model_config)
            infer_model = Model(data, model_config)
            infer_sess = tf.Session(config=tf_config, graph=graph)
            infer_sess.run(tf.global_variables_initializer())
            infer_sess.run(tf.local_variables_initializer())
            infer_sess.run(infer_model.data_iterator.initializer)
            infer_sess.run(tf.tables_initializer())

        infer_model.load(infer_sess, model_dir=best_model_dir)
        self.infer_preprocessor = infer_preprocessor
        return infer_model, infer_sess

    def build_graph(self, name="train"):
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with graph.as_default():
            self.logger.info("Building {} graph...".format(name))
            Model = get_model(self.config.model)
            model = Model(self.data, self.config)
            sess = tf.Session(config=tf_config, graph=graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(model.data_iterator.initializer)
            if (self.config.checkpoint_dir) and (name == "train"):
                self.logger.info('Loading checkpoint from {}'.format(
                    self.checkpoint_dir))
                model.load(sess)
                self.global_step = model.global_step_tensor.eval(sess)
        return model, sess

    def train_step(self, model, sess):
        
        feed_dict = {model.lstm_dropout_keep_prob: self.config.lstm_dropout_keep_prob,
                     model.num_negative_samples: self.config.num_negative_samples,
                     model.embed_dropout_keep_prob: self.config.embed_dropout_keep_prob,
                     model.add_echo: (self.config.add_echo) & (self.global_step > 100000)
                     }
        
        if self.use_weak_supervision:
            input_queries, input_replies, query_lengths, reply_lengths, weak_distances = \
            self.infer_sess.run([self.infer_model.input_queries, 
                                 self.infer_model.input_replies, 
                                 self.infer_model.queries_lengths, 
                                 self.infer_model.replies_lengths, self.infer_model.distances], 
                                feed_dict={self.infer_model.dropout_keep_prob: 1, 
                                           self.infer_model.add_echo: False})
            feed_dict.update({model.input_queries: input_queries, 
                              model.input_replies: input_replies, 
                              model.query_lengths: query_lengths, 
                              model.reply_lengths: reply_lengths, 
                              model.weak_distances: weak_distances})
        
            if (weak_distances.shape[0] != input_queries.shape[0]) and (weak_distances.shape[1] != input_queries.shape[0]):
                self.logger.info("Wrong Weak Distance!!!!")
                return None, None
        
        _, loss, score = sess.run([model.train_step, model.loss, model.accuracy],
                                  feed_dict=feed_dict)

        return loss, score

    def train_epoch(self, model, sess):
        """Not used because data size is too big"""
        self.cur_epoch += 1
        loop = tqdm(range(self.num_steps_per_epoch))
        losses = list()
        scores = list()

        for step in loop:
            loss, score = self.train_step(model, sess)
            losses.append(loss)
            scores.append(score)
        train_loss = np.mean(losses)
        train_score = np.mean(scores)

    def train(self):
        # build train, val graph
        train_model, train_sess = self.build_graph(name="train")
        val_model, val_sess = self.build_graph(name="val")

        # get global step and cur epoch from loaded model, zero if there is no loaded model
        self.global_step = train_model.global_step_tensor.eval(train_sess)
        self.cur_epoch = train_model.cur_epoch_tensor.eval(train_sess)

        for epoch in range(self.cur_epoch, self.config.num_epochs + 1, 1):
            self.logger.warn("=" * 35 + " Epoch {} Start ! ".format(epoch) + "=" * 35)
            self.cur_epoch = epoch

            # initialize loss and score
            losses = list()
            scores = list()

            for step in tqdm(range(1, self.num_steps_per_epoch + 1)):
                # skip trained batches
                if ((epoch - 1) * self.num_steps_per_epoch) + step < self.global_step:
                    continue

                loss, score = self.train_step(train_model, train_sess)
                if loss==None:
                    continue

                # increment global step
                self.global_step += 1
                train_sess.run(train_model.increment_global_step_tensor)

                # add loss and score
                losses.append(loss)
                scores.append(score)

                # summarize every 50 steps
                if self.global_step % 50 == 0:
                    self.summary_writer.summarize(self.global_step,
                                                  summarizer="train",
                                                  summaries_dict={"loss": np.array(loss),
                                                                  "score": np.array(score)})
                # save model
                if self.global_step % self.config.save_every == 0:
                    train_model.save(train_sess, os.path.join(self.checkpoint_dir, "model.ckpt"))

                # evaluate model
                if self.global_step % self.config.evaluate_every == 0:
                    val_loss, val_score = self.val(val_model, val_sess, self.global_step)
                    train_loss, train_score = np.mean(losses), np.mean(scores)
                    self.logger.warn(self.train_summary.format(self.cur_epoch, step, train_loss, train_score) \
                                     + self.val_summary.format(val_loss, val_score))
                    # initialize loss and score
                    losses = list()
                    scores = list()

            # val step
            self.logger.warn("=" * 35 + " Epoch {} Done ! ".format(epoch) + "=" * 35)
            train_model.save(train_sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            val_loss, val_score = self.val(val_model, val_sess, self.global_step)
            self.logger.warn(self.val_summary.format(val_loss, val_score))

            # increment epoch tensor
            train_sess.run(train_model.increment_cur_epoch_tensor)

    def val(self, model, sess, global_step):
        # load latest checkpoint
        model.load(sess)
        sess.run(model.data_iterator.initializer)

        # initialize loss and score
        losses = list()
        scores = list()

        # define loop
        num_batches_per_epoch = (self.data.val_size - 1) // self.batch_size + 1
        loop = tqdm(range(1, num_batches_per_epoch + 1))

        for step in loop:
            feed_dict = {model.lstm_dropout_keep_prob: 1,
                         model.num_negative_samples: 4,
                         model.embed_dropout_keep_prob: 1,
                         model.add_echo: False
                         }
            loss, score = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
            losses.append(loss)
            scores.append(score)
        val_loss = np.mean(losses)
        val_score = np.mean(scores)

        # summarize val loss and score
        self.summary_writer.summarize(global_step,
                                      summarizer="val",
                                      summaries_dict={"score": np.array(val_score),
                                                      "loss": np.array(val_loss)})

        # save as best model if it is best score
        best_loss = float(getattr(self.config, "best_loss", 1e+5))
        if val_loss < best_loss:
            self.logger.warn(
                "[Step {}] Saving for best loss : {:.5f} -> {:.5f}".format(global_step, best_loss, val_loss))
            model.save(sess,
                       os.path.join(self.checkpoint_dir, "best_loss", "best_loss.ckpt"))
            setattr(self.config, "best_loss", "{:.5f}".format(val_loss))
            # save best config
            setattr(self.config, "best_step", str(self.global_step))
            setattr(self.config, "best_epoch", str(self.cur_epoch))
            save_config(self.config.checkpoint_dir, self.config)
        return val_loss, val_score


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
        feed_dict = {
            model.lstm_dropout_keep_prob: self.config.lstm_dropout_keep_prob,
            model.num_negative_samples: self.config.num_negative_samples,
            model.embed_dropout_keep_prob: self.config.embed_dropout_keep_prob,
            model.dense_dropout_keep_prob: self.config.dense_dropout_keep_prob
        }

        _, loss, score, lr = sess.run(
            [model.train_step, model.loss, model.accuracy, model.learning_rate],
            feed_dict=feed_dict)

        return loss, score, lr

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
        # Build train, val graph
        train_model, train_sess = self.build_graph(name="train")
        val_model, val_sess = self.build_graph(name="val")

        # Get global step and cur epoch from loaded model, zero if there is no loaded model
        self.global_step = train_model.global_step_tensor.eval(train_sess)
        self.cur_epoch = train_model.cur_epoch_tensor.eval(train_sess)

        for epoch in range(self.cur_epoch, self.config.num_epochs + 1, 1):
            self.logger.warn("=" * 35 + " Epoch {} Start ! ".format(epoch) + "=" * 35)
            self.cur_epoch = epoch

            # Initialize loss and score
            losses = list()
            scores = list()

            for step in tqdm(range(1, self.num_steps_per_epoch + 1)):
                # Skip trained batches
                if ((epoch - 1) * self.num_steps_per_epoch) + step < self.global_step:
                    continue

                loss, score, lr = self.train_step(train_model, train_sess)

                # Increment global step
                self.global_step += 1
                train_sess.run(train_model.increment_global_step_tensor)

                # Add loss and score
                losses.append(loss)
                scores.append(score)

                # Summarize every 50 steps
                if self.global_step % 50 == 0:
                    self.summary_writer.summarize(
                        self.global_step,
                        summarizer="train",
                        summaries_dict={
                            "loss": np.array(loss),
                            "score": np.array(score),
                            "learning_rate": np.array(lr)
                        })

                # Save model
                if self.global_step % self.config.save_every == 0:
                    train_model.save(
                        train_sess, os.path.join(self.checkpoint_dir, "model.ckpt"))

                # Evaluate model
                if self.global_step % self.config.evaluate_every == 0:
                    val_loss, val_score = self.val(
                        val_model, val_sess, self.global_step)
                    train_loss, train_score = np.mean(losses), np.mean(scores)
                    self.logger.warn(
                        self.train_summary.format(self.cur_epoch,
                                                  step,
                                                  train_loss,
                                                  train_score) \
                        + self.val_summary.format(val_loss, val_score))

                    # Initialize loss and score
                    losses = list()
                    scores = list()

            # Validation step
            self.logger.warn("=" * 35 + " Epoch {} Done ! ".format(epoch) + "=" * 35)
            train_model.save(train_sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            val_loss, val_score = self.val(val_model, val_sess, self.global_step)
            self.logger.warn(self.val_summary.format(val_loss, val_score))

            # Increment epoch tensor
            train_sess.run(train_model.increment_cur_epoch_tensor)

    def val(self, model, sess, global_step):
        # Load latest checkpoint
        model.load(sess)
        sess.run(model.data_iterator.initializer)

        # Initialize loss and score
        losses = list()
        scores = list()

        # Define loop
        num_batches_per_epoch = (self.data.val_size - 1) // self.batch_size + 1
        loop = tqdm(range(1, num_batches_per_epoch + 1))

        for step in loop:
            feed_dict = {model.lstm_dropout_keep_prob: 1,
                         model.num_negative_samples: 4,
                         model.embed_dropout_keep_prob: 1,
                         model.dense_dropout_keep_prob: 1}
                
            loss, score = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
            losses.append(loss)
            scores.append(score)
        val_loss = np.mean(losses)
        val_score = np.mean(scores)

        # Summarize val loss and score
        self.summary_writer.summarize(global_step,
                                      summarizer="val",
                                      summaries_dict={"score": np.array(val_score),
                                                      "loss": np.array(val_loss)})

        # Save as best model if it is best score
        best_loss = float(getattr(self.config, "best_loss", 1e+5))
        if val_loss < best_loss:
            self.logger.warn(
                "[Step {}] Saving for best loss : {:.5f} -> {:.5f}".format(global_step,
                                                                           best_loss,
                                                                           val_loss))
            model.save(
                sess, os.path.join(self.checkpoint_dir, "best_loss", "best_loss.ckpt"))
            setattr(self.config, "best_loss", "{:.5f}".format(val_loss))

            # Save best config
            setattr(self.config, "best_step", str(self.global_step))
            setattr(self.config, "best_epoch", str(self.cur_epoch))
            save_config(self.config.checkpoint_dir, self.config)
        return val_loss, val_score

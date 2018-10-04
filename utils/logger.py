import tensorflow as tf
import os
import json
import logging
from colorlog import ColoredFormatter


class SummaryWriter:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.summary_dir = self.config.checkpoint_dir + "summaries/"
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag, value)
        :return:
        """
        writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))
                for summary in summary_list:
                    writer.add_summary(summary, step)
                writer.flush()


def setup_logger():
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s][%(levelname)s] %(message)s %(reset)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )
    logger = logging.getLogger('my_logger')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

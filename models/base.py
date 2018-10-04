import tensorflow as tf
import re
import os

from utils.logger import setup_logger

class Model:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()
        self.logger = setup_logger()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, save_dir):
        self.saver.save(sess, save_dir)

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, model_dir=""):
        if model_dir:
            self.saver.restore(sess, model_dir)
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(sess, latest_checkpoint)
            else:
                self.logger.error("No checkpoint found in {}".format(self.config.checkpoint_dir))
        
    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(1, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(1, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.config.max_to_keep, 
                                    keep_checkpoint_every_n_hours=1)

    def build_model(self):
        raise NotImplementedError

def get_model(model_name):
    underscored_model_name = camel_to_underscore(model_name)
    exec("from models.{} import {}".format(underscored_model_name, model_name))
    model = eval(model_name)
    return model
    
def camel_to_underscore(string):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

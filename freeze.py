"""Freeze trained model"""
import os
import json
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import argparse

from dataset import Dataset
from preprocessor import Preprocessor
from utils.logger import setup_logger
from utils.config import load_config
from models.base import get_model

args = argparse.ArgumentParser()
args.add_argument("--name", type=str, default="")
args.add_argument("--normalizer", type=str, default="MatchingModelNormalizer")
args.add_argument("--tokenizer", type=str, default="SentPieceTokenizer")
args.add_argument("--save_dir", type=str, default="")

def main():
    config = args.parse_args()
    logger = setup_logger()
    base_dir = "/media/scatter/scatterdisk/reply_matching_model/runs/{}/".format(
        config.name)
    model_config_path = base_dir + "config.json"
    model_path = base_dir + "best_loss/best_loss.ckpt"
    model_config = load_config(model_config_path)
    preprocessor = Preprocessor(model_config)

    graph = tf.Graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    logger.info("Loading model : {}".format(config.name))
    with graph.as_default():
        Model = get_model(model_config.model)
        data = Dataset(preprocessor,
                       model_config.train_dir,
                       model_config.val_dir,
                       model_config.min_length,
                       model_config.max_length,
                       model_config.num_negative_samples,
                       model_config.batch_size,
                       model_config.shuffle,
                       model_config.num_epochs,
                       debug=False)
        infer_model = Model(data, model_config, mode=tf.contrib.learn.ModeKeys.EVAL)
        infer_sess = tf.Session(config=tf_config, graph=graph)
        infer_sess.run(tf.global_variables_initializer())
        infer_sess.run(tf.local_variables_initializer())

    infer_model.load(infer_sess, model_dir=model_path)
    graph_path = os.path.join(config.save_dir, config.name)

    logger.info("Writing graph at {}".format(graph_path))
    tf.train.write_graph(infer_sess.graph, graph_path, "graph.pbtxt")
    logger.info("Done")

    maybe_output_node_names = ["positive_distances",
                               "queries_encoded",
                               "replies_encoded",
                               "probs",
                               "predictions"]
    delete = ["increment_global_step_tensor", 
              "increment_cur_epoch_tensor", 
              "learning_rate", 
              "labels", 
              "loss", 
              "accuracy", 
              "dense_dropout_keep_prob"]

    key2node_name = dict()
    output_node_names = []
    for name, att in vars(infer_model).items():
        if isinstance(att, tf.Tensor):
            key2node_name[name] = "import/" + att.name
            for node_name in maybe_output_node_names:
                if node_name in name:
                    output_node_names.append(att.name)
                    
    for trash in delete:
        if trash in key2node_name:
            del key2node_name[trash]
    
    json.dump(key2node_name,
              open(os.path.join(
                  config.save_dir, config.name, "key_to_node_name.json"), "w"))
    output_node_names = list(set([name.split(":")[0] for name in output_node_names]))

    logger.info("Freezing...")
    freeze_graph.freeze_graph(
        input_graph=os.path.join(graph_path, "graph.pbtxt"),
        input_saver="",
        input_binary=False,
        input_checkpoint=model_path,
        output_node_names=",".join(output_node_names),
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const:0",
        output_graph=os.path.join(config.save_dir, config.name, "model"),
        clear_devices=True,
        initializer_nodes='string_to_index/hash_table/table_init'
    )
    logger.info("Done.")

    
if __name__ == "__main__":
    main()

import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime

from dataset import Dataset
from trainer import MatchingModelTrainer
from preprocessor import Preprocessor
from utils.dirs import create_dirs
from utils.logger import SummaryWriter
from utils.config import load_config, save_config
from models.base import get_model
from utils.utils import JamoProcessor

now = datetime.now()

# Parameters
# ==================================================

# Task specification
args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="train", choices=["train", "debug", "infer"])
args.add_argument("--name", type=str, default="start")
args.add_argument("--config", type=str, default="")

# Data loading and saving parameters
args.add_argument("--train_dir", type=str, default="/media/scatter/scatterdisk/reply_matching_model/sol.tokenized.sent_piece_100K/")
args.add_argument("--val_dir", type=str, default="/media/scatter/scatterdisk/reply_matching_model/sol.tokenized.sent_piece_100K/sol.validation.txt")
args.add_argument("--pretrained_embed_dir", type=str, default="/media/scatter/scatterdisk/pretrained_embedding/sent_piece_100K/fasttext.sent_piece_100K.256D")
args.add_argument("--checkpoint_dir", type=str, default="/media/scatter/scatterdisk/reply_matching_model/runs/")

# Model specification
args.add_argument("--model", type=str, default="DualEncoderLSTM")
args.add_argument("--sent_piece_model", type=str, default="/media/scatter/scatterdisk/tokenizer/sent_piece.100K.model")
args.add_argument("--soynlp_scores", type=str, default="/media/scatter/scatterdisk/tokenizer/soynlp_scores.sol.100M.txt")
args.add_argument("--normalizer", type=str, default="DummyNormalizer")
args.add_argument("--tokenizer", type=str, default="DummyTokenizer")
args.add_argument("--vocab_size", type=int, default=90000)
args.add_argument("--vocab_list", type=str, default="/media/scatter/scatterdisk/pretrained_embedding/vocab_list.sent_piece_100K.txt")

# Data specification
args.add_argument("--batch_size", type=int, default=512)
args.add_argument("--num_epochs", type=int, default=10)
args.add_argument("--evaluate_every", type=int, default=20000)
args.add_argument("--save_every", type=int, default=20000)
args.add_argument("--max_to_keep", type=int, default=5)
args.add_argument("--shuffle", type=bool, default=True)

# Embedding, learning rate specification
args.add_argument("--embed_dim", type=int, default=256)
args.add_argument("--embed_dropout_keep_prob", type=float, default=0.9)
args.add_argument("--min_length", type=int, default=1)
args.add_argument("--max_length", type=int, default=30)
args.add_argument("--learning_rate", type=float, default=1e-3)

# Model : DualEncoderLSTM
args.add_argument("--lstm_dim", type=int, default=512)
args.add_argument("--lstm_dropout_keep_prob", type=float, default=0.9)
args.add_argument("--dense_dropout_keep_prob", type=float, default=0.9)

# Model : CNNMatch
args.add_argument("--filter_sizes", type=str, default="2,3")
args.add_argument("--num_filters", type=int, default=64)
args.add_argument("--num_hidden", type=int, default=128)
args.add_argument("--hidden_dropout_keep_prob", type=float, default=0.9)

args.add_argument("--hinge_loss", type=float, default=0.5)

# Weak Supervision
args.add_argument("--weak_supervision", type=bool, default=False)

# Sampling parameters
args.add_argument("--negative_sampling", type=str, default="random", choices=["random", "hard", "weighted"])
args.add_argument("--num_negative_samples", type=int, default=4)
args.add_argument("--add_echo", type=bool, default=False)

def main():
    config = args.parse_args()
    # Load pre-defined config if possible
    if config.config:
        config = load_config(config.config)

    # Print config
    config_str = " | ".join(["{}={}".format(attr.upper(), value) for attr, value in vars(config).items()])
    print(config_str)
    config = create_dirs(config)
    
    # Create tensorflow session
    device_config = tf.ConfigProto()
    device_config.gpu_options.allow_growth = True
    sess = tf.Session(config=device_config)

    # Build preprocessor
    preprocessor = Preprocessor(config)

    # Load data, preprocess and generate data
    data = Dataset(preprocessor, 
               config.train_dir, 
               config.val_dir, 
               config.min_length, 
               config.max_length,
               config.num_negative_samples,
               config.batch_size, 
               config.shuffle, 
               config.num_epochs, 
               debug=False)

    # Create tensorboard summary writer
    summary_writer = SummaryWriter(sess, config)

    # Create trainer and pass all the previous components to it
    trainer = MatchingModelTrainer(sess, preprocessor, data, config, summary_writer)

    # Here you train your model
    trainer.train()


if __name__ == '__main__':
    main()

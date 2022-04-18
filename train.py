"""Module for training an NLI InferSent model"""
import argparse

import torch
import pytorch_lightning as pl

from models.infersent import InferSent
import data
import utils


def train(args):
    """
    Trains an InferSent model.
    Test-set evaluation is deferred to eval.py
    """
    # set up logger, trainer and seed
    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.model_type,
    )
    lr_monitor = utils.HackedLearningRateMonitor(logging_interval="epoch")
    early_stopper = pl.callbacks.EarlyStopping(
        monitor="lr_log", patience=0, mode="min", stopping_threshold=1e-5
    )
    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        logger=logger,
        enable_progress_bar=args.progress_bar,
        callbacks=[lr_monitor, early_stopper],
    )
    pl.seed_everything(args.seed)  # for reproducibility
    # load data and setup data
    snli = data.SNLIDataModule(args.batch_size, args.data_dir, 4, args.cached_vocab)
    snli.prepare_data()
    snli.setup()
    # if we've provided pre-aligned glove embeddings tensor, then load directly
    if args.aligned_glove:
        aligned_glove = torch.load(args.aligned_glove)
    else:
        # otherwise we have to compute it from glove and our vocab
        glove = data.GloVe(args.glove, args.glove_variant)
        aligned_glove = data.align_glove_to_vocab(glove, snli.vocab)
    # if checkpoint provided, load model from it, otherwise initialize new model
    if args.checkpoint_path:
        model = InferSent.load_from_checkpoint(args.checkpoint_path)
        # glove embeddings are not saved in checkpoint so we have to load separately
        model.load_embeddings(aligned_glove)
    else:
        model = InferSent(args.encoder_type, snli.vocab, aligned_glove.shape[1])
        # overwrite randomly initialized embeddings with glove
        model.load_embeddings(aligned_glove)
    # first train
    trainer.fit(model, datamodule=snli)
    # evaluation on test set is handled by eval.py, where we also use SentEval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains an InferSent model. "
        "Test-set evaluation is deferred to eval.py"
    )
    parser.add_argument(
        "-e",
        "--encoder-type",
        type=str,
        help="one of 'baseline', 'lstm', 'bilstm', maxpoolbilstm'",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="path for loading previously saved checkpoint",
    )
    parser.add_argument(
        "-cd",
        "--checkpoint-dir",
        type=str,
        help="where to save checkpoints. Defaults to the current working directory",
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="the random seed to use", default=42
    )
    parser.add_argument(
        "-p",
        "--progress-bar",
        action="store_true",
        help="whether to show the progress bar",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        help="path to log directory",
        default="logs/",
    )
    parser.add_argument(
        "-d", "--data-dir", type=str, help="path to data directory", default="data/"
    )
    parser.add_argument(
        "-g",
        "--glove",
        type=str,
        help="path to glove embeddings",
        default="data/glove.840B.300d.txt",
    )
    parser.add_argument(
        "-gv",
        "--glove-variant",
        type=str,
        help="which variant of glove embeddings to use",
        default="840B300d",
    )
    parser.add_argument(
        "-ag",
        "--aligned-glove",
        help="path to aligned glove embeddings tensor",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="batch size for training",
        default=64,
    )
    parser.add_argument(
        "-cv",
        "--cached-vocab",
        help="path to save/load serialized vocabulary",
        type=str,
        default="data/cached_vocab.pkl",
    )

    args = parser.parse_args()

    train(args)

"""Module for training an NLI InferSent model"""
import argparse
import pytorch_lightning as pl
from models.infersent import InferSent
from data.snli import SNLIDataModule


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
    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        logger=logger,
        enable_progress_bar=args.progress_bar,
    )
    pl.seed_everything(args.seed)  # for reproducibility
    # if checkpoint provided, load model from it, otherwise initialize new model
    if args.checkpoint_path:
        model = InferSent.load_from_checkpoint(args.checkpoint_path)
    else:
        model = InferSent(args.encoder_type)
    # load data
    snli = SNLIDataModule(args.batch_size)
    # first train
    trainer.fit(model, datamodule=snli)
    # evaluation on test set is handled by eval.py, where we also use SentEval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains an InferSent model. "
        "Test-set evaluation is deferred to eval.py"
    )
    parser.add_argument(
        "--encoder_type",
        "-e",
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
        "-b",
        "--batch-size",
        type=int,
        help="batch size for training",
        default=64,
    )

    args = parser.parse_args()

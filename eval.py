import argparse
import pickle
import os
from warnings import simplefilter
import logging

import torch
import pytorch_lightning as pl
from models.infersent import InferSent
import senteval
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from data import Vocabulary, SNLIDataModule


def batcher(params, batch):
    """
    batcher function needed for SentEval
    """
    model = params["model"]
    sent_embs = np.array(
        [
            model.encoder.encode(" ".join(sentence) if len(sentence) > 0 else ".")
            .detach()
            .numpy()
            for sentence in batch
        ]
    ).squeeze(1)
    return sent_embs


def eval_senteval(args, model):
    """Evaluates a model on SentEval, serializing the results"""
    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)
    # using prototyping config from the SentEval github
    params = {"task_path": args.data_dir, "usepytorch": args.gpu, "kfold": 5}
    params["classifier"] = {
        "nhid": 0,
        "optim": "rmsprop",
        "batch_size": 128,
        "tenacity": 3,
        "epoch_size": 2,
    }
    # so that we can pass the model to the batcher function somehow
    params["model"] = model
    # initialize the SentEval engine
    se = senteval.engine.SE(params, batcher)
    # here are the tasks we want to evaluate on: same as the InferSent paper
    tasks = [
        "MR",
        "CR",
        "SUBJ",
        "MPQA",
        "SST2",
        "TREC",
        "MRPC",
        "SICKRelatedness",
        "SICKEntailment",
        "STS14",
    ]
    # do the evaluation
    results = se.eval(tasks)

    print("SentEval Evaluation complete. Saving results...")
    # create directory if it doesn't exist
    if not os.path.exists(args.senteval_output_dir):
        os.makedirs(args.senteval_output_dir)
    with open(os.path.join(args.senteval_output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


def eval_snli(args, model):
    """Evaluates a model on SNLI, serializing the results"""

    snli = SNLIDataModule(
        args.batch_size, args.data_dir, args.num_workers, args.cached_vocab
    )
    snli.prepare_data()
    snli.setup()

    trainer = pl.Trainer(
        devices="auto",
        logger=False,
        accelerator="auto",
        enable_progress_bar=args.progress_bar,
    )
    print("Evaluating model on SNLI: validation")
    val_results = trainer.validate(model, verbose=True, datamodule=snli)
    print("Evaluating model on SNLI: test")
    test_results = trainer.test(model, verbose=True, datamodule=snli)

    print("SNLI Evaluation complete. Saving results...")
    # create directory if it doesn't exist
    if not os.path.exists(args.snli_output_dir):
        os.makedirs(args.snli_output_dir)
    with open(os.path.join(args.snli_output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_results, f)
    with open(os.path.join(args.snli_output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_results, f)


def eval_total(args):
    """evaluates a model on SNLI and SentEval"""
    model = InferSent.load_from_checkpoint(args.checkpoint_path)
    aligned_glove = torch.load(args.aligned_glove)
    # glove embeddings are not saved in checkpoint so we have to load separately
    model.load_embeddings(aligned_glove)

    if args.snli:
        eval_snli(args, model)
    if args.senteval:
        eval_senteval(args, model)


if __name__ == "__main__":
    # ignore convergence warnings
    simplefilter("ignore", category=ConvergenceWarning)
    # defined args
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on SNLI and SentEval"
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="data",
        help="path to data directory",
    )
    parser.add_argument(
        "--snli", action="store_true", help="Evaluate on SNLI", default=False
    )
    parser.add_argument(
        "--snli-output-dir",
        type=str,
        help="Directory to save SNLI results",
        default="results/snli/",
    )
    parser.add_argument(
        "--senteval", action="store_true", help="Evaluate on SentEval", default=False
    )
    parser.add_argument(
        "--senteval-output-dir",
        type=str,
        help="Directory to save SentEval results",
        default="results/senteval/",
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
        help="path to the checkpoint file",
        required=True,
    )
    parser.add_argument(
        "-ag",
        "--aligned-glove",
        type=str,
        help="path to the aligned glove file",
        default="data/aligned_glove.pt",
    )
    parser.add_argument(
        "-cv",
        "--cached-vocab",
        help="path to save/load serialized vocabulary",
        type=str,
        default="data/cached_vocab.pkl",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        help="number of workers for data loading",
        default=4,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="batch size",
        default=64,
    )
    parser.add_argument(
        "-p",
        "--progress-bar",
        action="store_true",
        help="whether to show the progress bar",
        default=False,
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="the random seed to use", default=42
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="whether to use gpu", default=False
    )
    # parse args
    args = parser.parse_args()
    # set seed for reproducibility
    pl.seed_everything(args.seed)
    # and evaluate
    eval_total(args)

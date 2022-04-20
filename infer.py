"""Script for inference"""
import argparse

import torch

from models.infersent import InferSent
from data import Vocabulary


def infer_nli(model: InferSent, premise, hypothesis, map_pred):
    """
    Infer the entailment of a premise and hypothesis pair
    """
    return model.predict(premise, hypothesis, map_pred)


def infer_sentembed(model, sentence):
    """
    Infer the sentence embedding of a sentence
    """
    torch.set_printoptions(threshold=5000)
    return model.encoder.encode(sentence)


def infer_main(model, args):

    if args.mode == "nli":
        if args.sentence_2 is None:
            raise ValueError("Must provide a hypothesis if NLI'ing")
        return infer_nli(model, args.sentence_1, args.sentence_2, args.map)
    elif args.mode == "sentembed":
        return infer_sentembed(model, args.sentence_1)
    else:
        raise ValueError("Mode must be one of 'nli' or 'sentembed'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for inference")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        help="Mode for inference. One of 'nli' or 'sentembed'",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "-ag",
        "--aligned-glove",
        type=str,
        help="path to the aligned glove file",
        default="data/aligned_glove.pt",
    )
    parser.add_argument(
        "-s1",
        "--sentence-1",
        type=str,
        required=True,
        help="Sentence to embed if embedding, premise if NLI'ing",
    )
    parser.add_argument(
        "-s2",
        "--sentence-2",
        type=str,
        help="Hypothesis. Only required if NLI'ing",
    )
    parser.add_argument(
        "-map",
        "--map",
        action="store_true",
        default=False,
        help="Flag whether to return one of "
        "{'entailment', 'neutral', 'contradiction'} instead of {0, 1, 2}",
    )
    args = parser.parse_args()

    # load model
    model = InferSent.load_from_checkpoint(args.checkpoint_path)
    aligned_glove = torch.load(args.aligned_glove)
    # glove embeddings are not saved in checkpoint so we have to load separately
    model.load_embeddings(aligned_glove)
    # perform inference
    output = infer_main(model, args)
    print(output)

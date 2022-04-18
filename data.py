"""DataLzfoaders and helpers"""
import os
import zipfile
from typing import Optional
import argparse
from collections import defaultdict
import pickle

import numpy as np
import pytorch_lightning as pl
import datasets
from nltk.tokenize import word_tokenize
import torch
from tqdm import tqdm

from utils import download_url


class GloVe:
    """GloVe embeddings"""

    def __init__(self, path: str, variant="840B300d"):
        variant_map = {
            "6B50d": {
                "num_lines": 400000,
                "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            },
            "6B100d": {
                "num_lines": 400000,
                "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            },
            "6B200d": {
                "num_lines": 400000,
                "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            },
            "6B300d": {
                "num_lines": 400000,
                "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            },
            "42B300d": {
                "num_lines": 1917494,
                "url": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
            },
            "840B300d": {
                "num_lines": 2196017,
                "url": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
            },
            "27B25d": {
                "num_lines": 1193514,
                "url": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
            },
            "27B50d": {
                "num_lines": 1193514,
                "url": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
            },
            "27B100d": {
                "num_lines": 1193514,
                "url": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
            },
            "27B200d": {
                "num_lines": 1193514,
                "url": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
            },
        }
        assert variant in variant_map, "Variant {} not found".format(variant)
        self.dim = int(variant.split("B")[1][:-1])
        self.download_url = variant_map[variant]["url"]
        self.num_lines = variant_map[variant]["num_lines"]
        self.filepath = path
        self.filename = os.path.splitext(os.path.basename(path))[0]
        self.dir = os.path.dirname(path) + "/"
        self.embeddings = {}
        self.vocab = set()
        self.load()

    def download_and_unzip(self):
        """Download embeddings"""
        print("Downloading GloVe embeddings...")
        download_url(
            self.download_url,
            os.path.join(self.dir, self.filename + ".zip"),
        )
        print("Unzipping GloVe embeddings...")
        with zipfile.ZipFile(
            os.path.join(self.dir, self.filename + ".zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(self.dir)
        print("Done.")

    def load(self, download=True):
        """Load embeddings"""
        # check if filepath exists
        if not os.path.exists(self.filepath):
            if download:
                self.download_and_unzip()
            else:
                raise FileNotFoundError(
                    "GloVe embeddings not found at {}".format(self.filepath)
                )

        with open(self.filepath, "r") as f:
            print("Parsing GloVe embeddings...")
            for line in tqdm(f, total=self.num_lines):
                split = line.split(" ")
                word = split[0]
                self.vocab.add(word)
                embedding = np.array([float(x) for x in split[1:]])
                self.embeddings[word] = embedding


class Vocabulary:
    """
    Helper class for handling vocabulary
    """

    def __init__(self, pad="<pad>", unk="<unk>"):
        self.pad = pad
        self.unk = unk
        self.word2idx = defaultdict(
            self._default_dict_default, {self.pad: 0, self.unk: 1}
        )
        self.idx2word = {0: self.pad, 1: self.unk}
        self.num_words = 2

    def _default_dict_default(self):
        """
        need a module-level function for pickle to work
        https://stackoverflow.com/a/16439720/9889508
        """
        return 1

    def add_word(self, word: str):
        """
        Add word to vocabulary
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.num_words += 1

    def __len__(self):
        return self.num_words

    def __contains__(self, word: str):
        return word in self.word2idx


class SNLIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for SNLI dataset.
    """

    def __init__(
        self,
        batch_size,
        data_dir,
        num_workers=4,
        cached_vocab_path: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_classes = 3
        self.num_workers = num_workers
        self.cached_vocab_path = cached_vocab_path
        self.cached_vocab = False
        self._setup_vocab()
        self.has_setup = False

    def _setup_vocab(self):
        """
        Either initializes an empty vocabulary or loads it from a file
        """
        if self.cached_vocab_path is not None:
            if os.path.exists(self.cached_vocab_path):
                print(
                    "Loading cached vocabulary from {}".format(self.cached_vocab_path)
                )
                with open(self.cached_vocab_path, "rb") as f:
                    self.vocab = pickle.load(f)
                self.cached_vocab = True
            else:
                print(
                    "Cached vocabulary not found at {}".format(self.cached_vocab_path)
                )
                self.vocab = Vocabulary()
        else:
            self.vocab = Vocabulary()

    def prepare_data(self):
        """
        Prepare data for training and evaluation.
        """
        # download data
        datasets.load_dataset(path="snli", cache_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """
        set up data: preprocess and build vocabulary
        """
        if self.has_setup:
            return
        # get train and val splits
        self.train_data, self.val_data, self.test_data = datasets.load_dataset(
            path="snli",
            cache_dir=self.data_dir,
            split=("train", "validation", "test"),
        )
        # filter out non gold labels
        print("Filtering out non-gold labels...")
        self.train_data = self.train_data.filter(lambda row: row["label"] >= 0)
        self.val_data = self.val_data.filter(lambda row: row["label"] >= 0)
        self.test_data = self.test_data.filter(lambda row: row["label"] >= 0)
        # process (lowercase and tokenize), populate the vocab and save as inst vars
        print("Processing data and building Vocab...")
        self.train_data = self._process_data(self.train_data)
        self.val_data = self._process_data(self.val_data)
        self.test_data = self._process_data(self.test_data)
        # convert to torch
        self.train_data.set_format("torch", columns=["prem_idxs", "hypo_idxs", "label"])
        self.val_data.set_format("torch", columns=["prem_idxs", "hypo_idxs", "label"])
        self.test_data.set_format("torch", columns=["prem_idxs", "hypo_idxs", "label"])
        # remember that we've already setup
        self.has_setup = True

    def train_dataloader(self):
        """
        Returns a DataLoader for training.
        """
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for validation.
        """
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for testing.
        """
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def _process_sentence(self, sentence: str, build_vocab: bool):
        """
        lower-cases and tokenises a sentence, while updating the vocab
        then converts a sentence to sequence of indices
        """
        tokens = word_tokenize(sentence.lower())
        if build_vocab:
            for token in tokens:
                # add_word checks if the word is not already in the vocab
                self.vocab.add_word(token)
        indices = [self.vocab.word2idx[token] for token in tokens]
        return indices

    def _process_data(self, dataset: datasets.arrow_dataset.Dataset):
        """
        Converts a dataset of sentences to a dataset of lists of word indices
        By lowercasing and tokenizing the sentences while updating the vocab
        """
        return dataset.map(
            lambda row: {
                "prem_idxs": self._process_sentence(
                    row["premise"], not self.cached_vocab
                ),
                "hypo_idxs": self._process_sentence(
                    row["hypothesis"], not self.cached_vocab
                ),
            },
            load_from_cache_file=self.cached_vocab,
        ).remove_columns(["premise", "hypothesis"])

    def _collate_fn(self, batch):
        """
        Collate function for PyTorch DataLoader
        """
        labels, prem, hyp = zip(
            *[(el["label"], el["prem_idxs"], el["hypo_idxs"]) for el in batch]
        )
        # pad hypothesis and premise, keeping track of original lengths
        prem_lens = torch.LongTensor([len(el) for el in prem]).unsqueeze(1)
        prem = torch.nn.utils.rnn.pad_sequence(prem, batch_first=True, padding_value=0)
        hyp_lens = torch.LongTensor([len(el) for el in hyp]).unsqueeze(1)
        hyp = torch.nn.utils.rnn.pad_sequence(hyp, batch_first=True, padding_value=0)
        # return as tuple of premise, hypothesis and labels
        return (prem, prem_lens), (hyp, hyp_lens), torch.LongTensor(labels)


def align_glove_to_vocab(glove: GloVe, vocab: Vocabulary) -> torch.Tensor:
    """
    Aligns GloVe embeddings to a vocabulary
    such that only words in the vocabulary are kept.
    Returns a tensor of shape (vocab_size, embedding_dim)
    With the embeddings order matching the word2idx's in the vocabulary
    """
    print("Aligning GloVe embeddings to vocab...")
    # initialize
    embeddings = np.zeros((len(vocab), glove.dim), dtype=float)
    # align glove to vocabulary
    for word in tqdm(vocab.word2idx.keys(), total=vocab.num_words):
        if word in glove.embeddings:
            embeddings[vocab.word2idx[word]] = glove.embeddings[word]
    # set <unk> token to average of all embeddings
    embeddings[1] = np.mean(embeddings, axis=0)
    # pad token embedding is just a 0-valued vector, so we can leave it as is
    return torch.Tensor(embeddings)


def setup_data(args):
    """
    Sets up data by downloading things for the first time and
    serializing processed data to disk to avoid having to do it again.
    """
    # (download and) parse glove
    glove = GloVe(args.glove, args.glove_variant)
    # (download and) parse snli
    snli = SNLIDataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        cached_vocab_path=args.cached_vocab,
    )
    snli.prepare_data()
    snli.setup()
    # save vocab if we don't have it already
    if not snli.cached_vocab:
        print("Saving vocab to {}".format(args.cached_vocab))
        with open(args.cached_vocab, "wb") as f:
            pickle.dump(snli.vocab, f)
    # align glove to snli vocab
    aligned_glove = align_glove_to_vocab(glove, snli.vocab)
    print("Saving aligned GloVe embeddings to disk...")
    torch.save(aligned_glove, args.aligned_glove)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sets up data:"
        "downloads data and aligns GloVe embeddings to SNLI vocab."
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
        help="path to save aligned glove embeddings tensor",
        type=str,
        required=True,
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
    )
    args = parser.parse_args()
    setup_data(args)

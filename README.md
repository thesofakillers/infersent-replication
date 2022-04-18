# InferSent: A partial replication

This repository contains the code for a partial replication of Conneau et al.
(2017), _Supervised Learning of Universal Sentence Representations from Natural
Language Inference Data_.

Four different sentence encoders are implemented and trained in the "Generic NLI
training scheme" as described in the original paper. In particular:

1. _Baseline_: averaging word embeddings to obtain sentence representations
2. _LSTM_: applied on the word embeddings, where the last hidden state is
   considered as sentence representation.
3. _BiLSTM_: where the last hidden state of forward and backward layers are
   concatenated as the sentence representations.
4. _Max-Pool BiLSTM_: with max pooling applied to the concatenation of
   word-level hidden states from both directions to retrieve sentence
   representations

Evaluation is then done with [SentEval](https://aclanthology.org/L18-1269/).

## Requirements

Details such as python and package versions can be found in the generated
[pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock) files.

We recommend using an environment manager such as
[conda](https://docs.conda.io/en/latest/). After setting up your environment
with the correct python version, please proceed with the installation of the
required packages

For [poetry](https://python-poetry.org/) users, getting setup is as easy as
running

```terminal
poetry install
```

We also provide a [requirements.txt](requirements.txt) file for
[pip](https://pypi.org/project/pip/) users who do not wish to use poetry.. In
this case, simply run

```terminal
pip install -r requirements.txt
```

## Repository Structure

```bash
.
├── models                             # models folder
│   ├── __init__.py
│   ├── encoder.py                     # sentence encoder models
│   └── infersent.py                   # Generic NLI model
├── utils.py                           # for miscellaneous utils
├── data.py                            # for data loading and processing
├── train.py                           # for model training
├── eval.py                            # for model evaluation
├── infer.py                           # for model inference
├── demo.ipynb                         # demo jupyter notebook
├── pyproject.toml                     # repo metadata
├── poetry.lock
├── gen_pip_conda_files.sh             # script for gen. pip and conda files
└── README.md                          # you are here

```

## Data, Pretrained Embeddings and Models

The repository does not include the datasets and pretrained embeddings used to
train the models mentioned above, nor the trained model checkpoints themselves,
as these are inappropriate for git version control.

The datasets used are public and will be automatically downloaded when
necessary. The same applies to the model checkpoints, hosted on the
[Internet Archive](https://archive.org).

The public datasets and embeddings used are
[SNLI](https://nlp.stanford.edu/projects/snli/) and 840B-token 300-d
[GloVe](https://nlp.stanford.edu/projects/glove/) respectively. If users already
have these locally and do not wish to re-download them, simply move (or
symbolically link them) to a shared data directory, and then signal this
directory and the resulting paths in the arguments for the scripts.

## Usage

### Demo

The repository comes with a demo [Jupyter Notebook](https://jupyter.org/) that
allows users to load a trained model and run inference on different examples.

The notebook also provides an overview and analysis of the results.

For more fine-grained usage, please refer to the following sections.

### Data and Embeddings

When called directly, `data.py` script will take care of setting up data and
embedding requirements for you. In particular, it will

1. Download GloVe embeddings if the embeddings .txt file is not found.
2. Parse the embeddings .txt file.
3. Download the SNLI dataset if they are not already downloaded.
4. Process the SNLI dataset, building the vocabulary in the process.
5. Save the vocab to disk, to avoid having to build it again.
6. Align GloVe embeddings to the vocab.
7. Save the aligned glove embeddings as a Tensor to disk.

For usage:

```stdout
usage: data.py [-h] [-d DATA_DIR] [-g GLOVE] [-gv GLOVE_VARIANT] -ag
               ALIGNED_GLOVE [-b BATCH_SIZE] [-cv CACHED_VOCAB]

Sets up data: downloads data and aligns GloVe embeddings to SNLI vocab.

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data-dir DATA_DIR
                        path to data directory
  -g GLOVE, --glove GLOVE
                        path to glove embeddings
  -gv GLOVE_VARIANT, --glove-variant GLOVE_VARIANT
                        which variant of glove embeddings to use
  -ag ALIGNED_GLOVE, --aligned-glove ALIGNED_GLOVE
                        path to save aligned glove embeddings tensor
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for training
  -cv CACHED_VOCAB, --cached-vocab CACHED_VOCAB
                        path to save/load serialized vocabulary
```

### Training

We use `train.py` for training. Most arguments should work fine with default
values.

For usage:

```stdout
usage: train.py [-h] -e ENCODER_TYPE [-c CHECKPOINT_PATH] [-cd CHECKPOINT_DIR]
                [-s SEED] [-p] [-l LOG_DIR] [-d DATA_DIR] [-g GLOVE]
                [-gv GLOVE_VARIANT] [-ag ALIGNED_GLOVE] [-b BATCH_SIZE]

Trains an InferSent model. Test-set evaluation is deferred to eval.py

options:
  -h, --help            show this help message and exit
  -e ENCODER_TYPE, --encoder-type ENCODER_TYPE
                        one of 'baseline', 'lstm', 'bilstm', maxpoolbilstm'
  -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        path for loading previously saved checkpoint
  -cd CHECKPOINT_DIR, --checkpoint-dir CHECKPOINT_DIR
                        where to save checkpoints. Defaults to the current
                        working directory
  -s SEED, --seed SEED  the random seed to use
  -p, --progress-bar    whether to show the progress bar
  -l LOG_DIR, --log-dir LOG_DIR
                        path to log directory
  -d DATA_DIR, --data-dir DATA_DIR
                        path to data directory
  -g GLOVE, --glove GLOVE
                        path to glove embeddings
  -gv GLOVE_VARIANT, --glove-variant GLOVE_VARIANT
                        which variant of glove embeddings to use
  -ag ALIGNED_GLOVE, --aligned-glove ALIGNED_GLOVE
                        path to aligned glove embeddings tensor
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for training

```

### Inference

TODO

### Evaluation

TODO

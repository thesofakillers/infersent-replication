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

## Requirements and Setup

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
[pip](https://pypi.org/project/pip/) users who do not wish to use poetry. In
this case, simply run

```terminal
pip install -r requirements.txt
```

Once these packages have been installed, we have to manually set up
[SentEval](https://github.com/facebookresearch/SentEval), since the FaceBook
researchers and engineers are not paid enough to make a PyPI package like
everyone else. To do this, first clone the repository to a folder of your
choice:

```terminal
git clone git@github.com:facebookresearch/SentEval.git
```

Then navigate to the SentEval repository and install it to the same environment
as used above:

```terminal
cd SentEval
python setup install
```

### Data, Pretrained Embeddings and Models

The repository does not include the datasets and pretrained embeddings used to
train the models mentioned above, nor the trained model checkpoints themselves,
as these are inappropriate for git version control.

The datasets relative to NLI training are public and will be automatically
downloaded when necessary.

The model checkpoints and evaluation results, hosted on the
[Internet Archive](https://archive.org), can be downloaded from
[this link](https://archive.org/download/thesofakillers-infersent-logs/logs.zip).
Please download and unzip the file, placing the resulting `logs/` directory in
the repository root.

The public datasets and embeddings used are
[SNLI](https://nlp.stanford.edu/projects/snli/) and 840B-token 300-d
[GloVe](https://nlp.stanford.edu/projects/glove/) respectively. If users already
have these locally and do not wish to re-download them, simply move (or
symbolically link them) to a shared data directory, and then signal this
directory and the resulting paths in the arguments for the scripts.

We also make use of the SentEval datasets. To download them, visit the senteval
repository you previously cloned and run

```terminal
cd .data/downstream/
./get_transfer_data.bash
```

Once this is complete, you may then rsync or mv the `downstream/` directory to a
directory of choice. Keep this directory in mind as we will then point to it
when using SentEval for evaluation. For example

```terminal
rsync -r -v -h senteval/data/downstream infersent-replication/data/
```

we would then point to `infersent-replication/data` when using SentEval for
evaluation

## Repository Structure

```bash
.
????????? models/                            # models folder
??????? ????????? __init__.py
??????? ????????? encoder.py                     # sentence encoder models
??????? ????????? infersent.py                   # Generic NLI model
????????? data/                              # data folder (not committed)
????????? logs/                              # logs folder (not committed)
????????? utils.py                           # for miscellaneous utils
????????? data.py                            # for data loading and processing
????????? train.py                           # for model training
????????? eval.py                            # for model evaluation
????????? infer.py                           # for model inference
????????? demo.ipynb                         # demo jupyter notebook
????????? error_analysis.md                  # error analysis markdown file
????????? error_analysis.pdf                 # error analysis pdf file
????????? images/                            # images folder for error_analysis.md
????????? pyproject.toml                     # repo metadata
????????? poetry.lock
????????? gen_pip_reqs.sh                    # script for gen. pip requirements
????????? README.md                          # you are here

```

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
usage: train.py [-h] -e ENCODER_TYPE [-c CHECKPOINT_PATH] [-s SEED] [-p]
                [-l LOG_DIR] [-d DATA_DIR] [-g GLOVE] [-gv GLOVE_VARIANT]
                [-ag ALIGNED_GLOVE] [-b BATCH_SIZE] [-cv CACHED_VOCAB]
                [-w NUM_WORKERS]

Trains an InferSent model. Test-set evaluation is deferred to eval.py

options:
  -h, --help            show this help message and exit
  -e ENCODER_TYPE, --encoder-type ENCODER_TYPE
                        one of 'baseline', 'lstm', 'bilstm', maxpoolbilstm'
  -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        path for loading previously saved checkpoint
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
  -cv CACHED_VOCAB, --cached-vocab CACHED_VOCAB
                        path to save/load serialized vocabulary
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        number of workers for data loading
```

### Evaluation

We use `eval.py` for evaluation, both on the original SNLI task as well as on
SentEval, configurably via the command-line arguments.

For usage:

```stdout
usage: eval.py [-h] [-d DATA_DIR] [-o OUTPUT_DIR] [--snli]
               [--snli-output-dir SNLI_OUTPUT_DIR] [--senteval]
               [--senteval-output-dir SENTEVAL_OUTPUT_DIR] -e ENCODER_TYPE -c
               CHECKPOINT_PATH [-ag ALIGNED_GLOVE] [-cv CACHED_VOCAB]
               [-w NUM_WORKERS] [-b BATCH_SIZE] [-p] [-s SEED] [-g]

Evaluate a trained model on SNLI and SentEval

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data-dir DATA_DIR
                        path to data directory
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Parent directory for saving results
  --snli                Evaluate on SNLI
  --snli-output-dir SNLI_OUTPUT_DIR
                        Directory to save SNLI results
  --senteval            Evaluate on SentEval
  --senteval-output-dir SENTEVAL_OUTPUT_DIR
                        Directory to save SentEval results
  -e ENCODER_TYPE, --encoder-type ENCODER_TYPE
                        one of 'baseline', 'lstm', 'bilstm', maxpoolbilstm'
  -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        path to the checkpoint file
  -ag ALIGNED_GLOVE, --aligned-glove ALIGNED_GLOVE
                        path to the aligned glove file
  -cv CACHED_VOCAB, --cached-vocab CACHED_VOCAB
                        path to save/load serialized vocabulary
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        number of workers for data loading
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  -p, --progress-bar    whether to show the progress bar
  -s SEED, --seed SEED  the random seed to use
  -g, --gpu             whether to use gpu
```

### Inference

We provide a simple script for performing inference, `infer.py`. This can be
used either to predict the entailment of a pair of sentences, or to embed a
particular sentence. For usage:

```stdout
usage: infer.py [-h] -m MODE -c CHECKPOINT_PATH [-ag ALIGNED_GLOVE] -s1
                SENTENCE_1 [-s2 SENTENCE_2] [-map]

Script for inference

options:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Mode for inference. One of 'nli' or 'sentembed'
  -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Path to the checkpoint file
  -ag ALIGNED_GLOVE, --aligned-glove ALIGNED_GLOVE
                        path to the aligned glove file
  -s1 SENTENCE_1, --sentence-1 SENTENCE_1
                        Sentence to embed if embedding, premise if NLI'ing
  -s2 SENTENCE_2, --sentence-2 SENTENCE_2
                        Hypothesis. Only required if NLI'ing
  -map, --map           Flag whether to return one of {'entailment',
                        'neutral', 'contradiction'} instead of {0, 1, 2}
```

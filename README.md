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

For [poetry](https://python-poetry.org/) users, getting setup is as easy as
running

```terminal
poetry install
```

We also provide an [environment.yml](environment.yml) file for
[Conda](https://docs.conda.io/projects/conda/en/latest/index.html) users who do
not wish to use poetry. In this case simply run

```terminal
conda env create -f environment.yml
```

Finally, if neither of the above options are desired, we also provide a
[requirements.txt](requirements.txt) file for
[pip](https://pypi.org/project/pip/) users. In this case, simply run

```terminal
pip install -r requirements.txt
```

## Repository Structure

```bash
.
├── models                             # models folder
│   ├── __init__.py
│   ├── baseline.py                       # model 1
│   ├── lstm.py                           # model 2
│   ├── bilstm.py                         # model 3
│   ├── maxpoolbilstm.py                  # model 4
│   └── nli.py                            # Generic NLI model
├── data.py                            # for data loading and processing
├── train.py                           # for model training
├── eval.py                            # for model evaluation
├── infer.py                           # for model inference
├── demo.ipynb                         # demo jupyter notebook
├── pyproject.toml                     # repo metadata
├── poetry.lock
└── README.md                          # you are here

```

## Data and Models

The repository does not include the datasets used to train the models mentioned
above, nor the trained model checkpoints themselves, as these are inappropriate
for git version control.

The datasets used are public and will be automatically downloaded when
necessary, if not already downloaded. The same applies to the model checkpoints,
hosted on the [Internet Archive](https://archive.org).

## Usage

TODO

### Demo

The notebook comes with a demo [Jupyter Notebook](https://jupyter.org/) that
allows users to load a trained model and run inference on different examples.

The notebook also provides an overview and analysis of the results.

For more fine-grained usage, please refer to the following sections.

### Training

TODO

### Inference

TODO

### Evaluation

TODO

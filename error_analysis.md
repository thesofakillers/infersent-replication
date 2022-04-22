---
title: ATCS Assignment 1 - Error Analysis
author: Giulio Starace
---

<!--
To compile to pdf, run:

pandoc error_analysis.md --shift-heading-level-by=-1 --highlight-style tango -F pandoc-crossref -C -o error_analysis.pdf
-->

![Training (left) and Validation (right) accuracy (bottom) and loss (top)
curves of the four models implemented.](images/learning_curves.png){#fig:learning_curves}

A few observations can be made with regards to the learning curves shown in
@Fig:learning_curves. First, from the validation accuracy curves we can see a
clear difference in performance between the baseline model and the LSTM models,
with the Max-pooled BiLSTM achieving the highest validation accuracy throughout.
From these curves it is interesting to note the similarity in performance
between the LSTM and the 'plain' BiLSTM, which actually does marginally worse
albeit training for slightly less epochs. It is unclear why this occurs,
especially since the BiLSTM contains the same information as the LSTM and could
just learn to discard the additional concatenated information from the reversed
LSTM. This may be explained by overfitting, with further evidence from the
BiLSTM training accuracy being higher than the LSTM. This, combined with the
much slower training time (2.05 hours vs 3.895 hours), indicates that the
'plain' BiLSTM is a worse choice than the LSTM, at least when it comes to SNLI
(we will see and discuss SentEval performance later).

On the subject of training time, we can gather more insights by hovering over
the curves. While we see that the implemented early stopping criterion (stop
when learning rate goes under $10^{-5}$) causes all models to end training
between 12-15 epochs, the underlying run time is quite different, ranging from
26 minutes for the baseline model to 3.992 hours for the MaxPoolBiLSTM model.
This range can be appreciated by changing the x-axis to relative.

One may argue that the stopping criterion outlined in the paper was a bit too
lenient and users may have benefited time-wise from more informed criterions.
The behaviour of the validation loss curves in the LSTM-based models, with an
initial dip followed by a slow rise may have been useful, particularly in
combination with the validation accuracy curves. Perhaps the authors intuited
that longer training could lead to better generalized sentence encoders at the
expense of slightly worse NLI performance. This is similar reasoning as for why
the same encoder is used for hypothesis and premise as opposed to using two
separate specialized encoders.

![Validation Loss (left) and Validation Accuracy of two slightly different MaxPoolBiLSTM
models](images/maxpoolbilstm_l_curves.png)

One final observation can be made from @fig:maxpoolbilstm_l_curves runs (version
1 and 3). Here, version 1 is an "incorrect" implementation of the MaxPoolBiLSTM
described by the authors, where the padded values of 0 hidden states the LSTM
are not masked out as they should be, to avoid incorrectly selecting them when
max-pooling hidden states with only negative values. Despite this incorrectness,
this version seems to outperform the "correct" implementation in terms of
validation accuracy. One possible explanation for this is that not masking out
the 0's somewhat simulates the application of a ReLU non-linearity, which
generally makes networks more expressive. Another explanation can be that the
zeros could mimic some kind of frankensteined dropout. Regardless, the
difference in performance is somewhat marginal and for the moment devoid of an
estimated uncertainty, making it difficult to rigorously compare. Because we
were interesting in replicating parts of the original work, the `version_1` was
disregarded for the rest of the work.

For further evaluation, we test our models both on the original SNLI task, as
well as the SentEval sentence embedding evaluation suite, particularly the 'MR',
'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', and 'SICKEntailment' transfer
tasks. This is done via `eval.py`. After some parsing of the results, we report
"micro" and "macro" validation set accuracy on the transfer tasks, where "macro"
refers to the classical average accuracy across the tasks while "micro" refers
to the average accuracy across the tasks weighted by the number of sample

|                | **dim** | **NLI**  |          | **Transfer** |           |
| :------------: | :-----: | :------: | :------: | :----------: | :-------: |
|                |         | **dev**  | **test** |  **micro**   | **macro** |
|   **Model**    |         |          |          |              |           |
|  **Baseline**  |   300   |   65.7   |   65.6   |     80.5     |   79.2    |
|    **LSTM**    |  2048   |   81.1   |   80.9   |     77.2     |   76.5    |
|   **BiLSTM**   |  4096   |   80.7   |   80.7   |     79.5     |   79.0    |
| **BiLSTM-Max** |  4096   | **84.3** | **84.2** |   **81.2**   | **80.7**  |

: Partial Replication of Table 3 of Conneau et al. {#tbl:table_3}

@tbl:table_3 shows the validation (a.k.a. "dev") and test accuracy of the four
models implemented in this repository on the NLI task, as evaluated on the SNLI
dataset. It also shows the micro and macro validation accuracy across the
SentEval tasks outlined above. A few remarks can be made.

Firstly, we see that the trends from validation accuracy on SNLI are mirrored in
the test accuracy, so the discussion from the validation curves above holds.
With regards to replication, for the LSTM and BiLSTM-Max model test accuracy we
are decently close to the original results of the authors.

What is perhaps more interesting however is how the models perform micro- and
macro-wise on the SentEval transfer tasks. We see that while the BiLSTM-Max
model still achieves the highest performance, the range across the various
architectures is now much more compact, ranging from 79.0 to 80.7 macro-wise and
77.2 to 81.2 micro-wise. This seems to suggest that the underlying force
dominating sentence-embedding performance is some aspect of the architecture
that is shared across all variants. This theory may explain why the Baseline
model, almost entirely based on the GloVe word-embeddings used in all models,
performs so comparatively well in this case, coming second only to the best
model both micro- and macro-wise.

If this were indeed the case, it would suggest that word-order is not properly
learned in the LSTM models, despite the sequential nature of their learning. We
can verify this by trying two sentences whose meanings are opposite but words
are the same. If word order matters, the resulting word-embeddings should
differ, i.e. the difference should be non-zero.

While the difference is indeed non-zero, we do note that the embeddings are
quite similar despite the sentences having opposite meanings. This can be
interpreted as word order not fully being exploited.

This may be a limit of the models employed, which at best visit sentences
sequentially in both directions, but cannot examine the sentences as a graph to
construct inner representations akin to a parse tree, where word order becomes
increasingly useful. At the expense of triteness, it would be interesting to
extend the InferSent architecture with a transformer, BERT-like encoder for the
sentences, to examine whether self-attention could be leveraged for improved
SentEval performance.

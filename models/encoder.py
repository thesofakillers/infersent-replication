import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize


class Encoder(nn.Module):
    """Base encoder class other encoders inherit from"""

    def __init__(self, vocab, word_emb_dim):
        super(Encoder, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.vocab = vocab
        self.embeddings = nn.Embedding(vocab.num_words, word_emb_dim)
        self.embeddings.weight.requires_grad = False
        # needs to be implemented in subclasses
        self.out_dim = None

    def load_embeddings(self, embeddings):
        """Load embeddings from a torch tensor"""
        self.embeddings.weight = torch.nn.Parameter(embeddings, requires_grad=False)

    def forward(self, sent_tuple):
        """Forward pass: to be implemented in subclasses"""
        raise NotImplementedError

    def encode(self, sentence: str):
        """Encode a sentence into a vector"""
        with torch.no_grad():
            tokens = word_tokenize(sentence.lower())
            indices = torch.LongTensor(
                [self.vocab.word2idx[token] for token in tokens]
            ).unsqueeze(0)
            lens = torch.LongTensor([len(tokens)]).unsqueeze(0)
            return self.forward((indices, lens))


class Baseline(Encoder):
    """Baseline encoder"""

    def __init__(self, vocab, word_emb_dim):
        super(Baseline, self).__init__(vocab, word_emb_dim)
        self.out_dim = word_emb_dim

    def forward(self, sent_tuple):
        """Forward pass"""
        # sent is B x L, where L is max(sent_len), B is batch size; sent_len is B x 1
        sent, sent_len = sent_tuple
        # embeddings is B x L x E, E is self.emb_dim
        emb = self.embeddings(sent)
        # out is B X E
        out = emb.sum(dim=1) / sent_len
        return out

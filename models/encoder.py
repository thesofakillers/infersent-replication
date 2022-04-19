from typing import Union, List

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

    def encode(self, sentence: Union[str, List[str]], tokenized: bool = False):
        """Encode a sentence into a vector"""
        with torch.no_grad():
            if not tokenized:
                tokens = word_tokenize(sentence.lower())
            else:
                tokens = [token.lower() for token in sentence]
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
        # embeddings is B x L x E, E is self.word_emb_dim
        emb = self.embeddings(sent)
        # out is B X E
        out = emb.sum(dim=1) / sent_len
        return out


class LSTM(Encoder):
    """LSTM encoder"""

    def __init__(self, vocab, word_emb_dim):
        super(LSTM, self).__init__(vocab, word_emb_dim)
        self.hid_dim = 2048
        self.out_dim = self.hid_dim
        self.lstm = nn.LSTM(word_emb_dim, self.hid_dim, batch_first=True)

    def forward(self, sent_tuple):
        """
        Forward pass. Need to pack.
        We are interested in the final hidden state of the LSTM.
        """
        # sent is B x L, where L is max(sent_len), B is batch size; sent_len is B x 1
        sent, sent_len = sent_tuple
        # embeddings is B x L x E, E is self.word_emb_dim
        emb = self.embeddings(sent)
        # packing for LSTM
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, sent_len.squeeze(1).to("cpu"), batch_first=True, enforce_sorted=False
        )
        # h_t is 1 X B x H, where H is self.out_dim
        _out, (h_t, _c_t) = self.lstm(packed_emb)
        return h_t.squeeze(0)


class BiLSTM(Encoder):
    """BiLSTM encoder"""

    def __init__(self, vocab, word_emb_dim):
        super(BiLSTM, self).__init__(vocab, word_emb_dim)
        self.hid_dim = 2048
        self.out_dim = 4096
        self.bilstm = nn.LSTM(
            word_emb_dim, self.hid_dim, batch_first=True, bidirectional=True
        )

    def forward(self, sent_tuple):
        """
        Forward pass. Need to pack.
        We are interested in the final hidden state of the BiLSTM.
        """
        # sent is B x L, where L is max(sent_len), B is batch size; sent_len is B x 1
        sent, sent_len = sent_tuple
        # embeddings is B x L x E, E is self.word_emb_dim
        emb = self.embeddings(sent)
        # packing for LSTM
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, sent_len.squeeze(1).to("cpu"), batch_first=True, enforce_sorted=False
        )
        # h_t is 2 X B x H, where H is self.hid_dim
        _out, (h_t, _c_t) = self.bilstm(packed_emb)
        # we concatenate the final hidden states of the forward and backward LSTMs
        out = torch.cat((h_t[0], h_t[1]), dim=1)
        return out


class MaxPoolBiLSTM(Encoder):
    """MaxPoolBiLSTM encoder"""

    def __init__(self, vocab, word_emb_dim):
        super(MaxPoolBiLSTM, self).__init__(vocab, word_emb_dim)
        self.hid_dim = 2048
        self.out_dim = 4096
        self.bilstm = nn.LSTM(
            word_emb_dim, self.hid_dim, batch_first=True, bidirectional=True
        )

    def forward(self, sent_tuple):
        """
        Forward pass. Need to pack.
        We are interested in the final hidden state of the BiLSTM.
        """
        # sent is B x L, where L is max(sent_len), B is batch size; sent_len is B x 1
        sent, sent_len = sent_tuple
        # embeddings is B x L x E, E is self.word_emb_dim
        emb = self.embeddings(sent)
        # packing for LSTM
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, sent_len.squeeze(1).to("cpu"), batch_first=True, enforce_sorted=False
        )
        # out is h_t at each t. shape is B x L x 2 * H, where H is self.hid_dim
        out_packed, (_h_t, _c_t) = self.bilstm(packed_emb)
        # to take maxpool, need to reapply padding
        out_padded, _sent_lens = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True
        )
        # we take maxpool across L to get (B x 2 * H), but need to ignore 0s
        with torch.no_grad():
            out_padded[out_padded == 0] = -1e9
        out, _max_idxs = torch.max(out_padded, dim=1)
        return out

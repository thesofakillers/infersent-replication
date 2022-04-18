"""Generic InferSent model"""
import torch
import pytorch_lightning as pl
import models.encoder as encoder
from nltk.tokenize import word_tokenize


class InferSent(pl.LightningModule):
    """InferSent model"""

    def __init__(self, encoder_type, vocab, word_emb_dim=300, hidden_dim=512):
        super().__init__()
        self.save_hyperparameters()
        self._set_encoder(encoder_type, vocab, word_emb_dim)
        self.inf2label = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }
        self.label2inf = {v: k for k, v in self.inf2label.items()}
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * self.encoder.out_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, prem_tuple, hyp_tuple):
        """Forward pass"""
        u = self.encoder(prem_tuple)
        v = self.encoder(hyp_tuple)

        relations = torch.cat((u, v, torch.abs(u - v), u * v), dim=1)

        out = self.mlp(relations)
        return out

    def _set_encoder(self, encoder_type, vocab, emb_dim):
        """Initializes and sets specific encoder"""
        assert encoder_type in [
            "baseline",
            "lstm",
            "bilstm",
            "maxpoolbilstm",
        ], "Invalid encoder type: {}".format(encoder_type)
        if encoder_type == "baseline":
            self.encoder = encoder.Baseline(vocab, emb_dim)
        elif encoder_type == "lstm":
            self.encoder = encoder.LSTM(vocab, emb_dim)
        elif encoder_type == "bilstm":
            self.encoder = encoder.BiLSTM(vocab, emb_dim)
        elif encoder_type == "maxpoolbilstm":
            self.encoder = encoder.MaxPoolBiLSTM(vocab, emb_dim)

    def _shared_step(self, batch):
        """Passes batch through model and computes loss and accuracy"""
        premise, hypothesis, y_true = batch
        y_pred = self.forward(premise, hypothesis)
        loss = torch.nn.functional.cross_entropy(y_pred, y_true)
        acc = torch.mean((y_pred.argmax(dim=1) == y_true).float())
        return loss, acc

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, acc = self._shared_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, acc = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Handles optimizers and schedulers"""
        # "we use SGD with a learning rate of 0.1"
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max", factor=0.2, patience=0
        )
        return [optimizer], [
            {
                "scheduler": step_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
            {
                "scheduler": plateau_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_acc",
            },
        ]

    def load_embeddings(self, embeddings):
        """loads pretrained embeddings"""
        self.encoder.load_embeddings(embeddings)

    def on_save_checkpoint(self, checkpoint):
        """Modifies checkpoint before saving by removing word-embeddings"""
        del checkpoint["state_dict"]["encoder.embeddings.weight"]

    def on_load_checkpoint(self, checkpoint):
        """Loaded checkpoint is missing the word-embeddings, so we init them again"""
        checkpoint["state_dict"]["encoder.embeddings.weight"] = torch.nn.Parameter(
            torch.randn(self.encoder.vocab.num_words, self.encoder.word_emb_dim),
            requires_grad=False,
        )

    def predict(self, premise: str, hypothesis: str, map_pred: bool = True):
        """Predicts entailment for given premise and hypothesis"""
        with torch.no_grad():
            # prepare premise input
            prem_tokens = word_tokenize(premise.lower())
            prem_idxs = torch.LongTensor(
                [self.encoder.vocab.word2idx[token] for token in prem_tokens]
            ).unsqueeze(0)
            prem_len = torch.LongTensor([len(prem_tokens)]).unsqueeze(0)
            prem_tuple = (prem_idxs, prem_len)
            # prepare hypothesis input
            hyp_tokens = word_tokenize(hypothesis.lower())
            hyp_idxs = torch.LongTensor(
                [self.encoder.vocab.word2idx[token] for token in hyp_tokens]
            ).unsqueeze(0)
            hyp_len = torch.LongTensor([len(hyp_tokens)]).unsqueeze(0)
            hyp_tuple = (hyp_idxs, hyp_len)
            # predict label
            y_pred = self.forward(prem_tuple, hyp_tuple)
            label = y_pred.argmax(dim=1).item()
            if map_pred:
                return self.label2inf[label]
            else:
                return label

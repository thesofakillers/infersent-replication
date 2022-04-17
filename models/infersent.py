"""Generic InferSent model"""
import torch
import pytorch_lightning as pl
import encoder


class InferSent(pl.LightningModule):
    """Generic InferSent model"""

    def __init__(self, encoder_type, vocab, num_classes=3, hidden_dim=512):
        super().__init__()
        self._set_encoder(encoder_type, vocab)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.loss = torch.nn.CrossEntropyLoss()
        self.encoder_dim = self.encoder.encoder_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * self.encoder_dim, self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, premise, hypothesis):
        """Forward pass"""
        u = self.encoder(premise)
        v = self.encoder(hypothesis)

        relations = torch.cat((u, v, torch.abs(u - v), u * v), dim=1)

        out = self.mlp(relations)
        return out

    def _set_encoder(self, encoder_type, vocab):
        """Initializes and sets specific encoder"""
        assert encoder_type in [
            "baseline",
            "lstm",
            "bilstm",
            "maxpoolbilstm",
        ], "Invalid encoder type: {}".format(encoder_type)
        if encoder_type == "baseline":
            self.encoder = encoder.Baseline(vocab)
        elif encoder_type == "lstm":
            self.encoder = encoder.LSTM(vocab)
        elif encoder_type == "bilstm":
            self.encoder = encoder.BiLSTM(vocab)
        elif encoder_type == "maxpoolbilstm":
            self.encoder = encoder.MaxPoolBiLSTM(vocab)

    def _shared_step(self, batch):
        """Passes batch through model and computes loss and accuracy"""
        premise, hypothesis, y_true = batch
        y_pred = self.forward(premise, hypothesis)
        loss = self.loss(y_pred, y_true)
        acc = torch.mean((y_pred.argmax(dim=1) == y_true).float())
        return loss, acc

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, acc = self._shared_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def val_step(self, batch, batch_idx):
        """Validation step"""
        loss, acc = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Handles optimizers and schedulers"""
        # "we use SGD with a learning rate of 0.1"
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        chained_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                # "and a weight decay of 0.99"
                torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99),
                # "we divide the learning rate by 5 if the dev accuracy decreases"
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, mode="max", factor=0.2, patience=0
                ),
            ]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": chained_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_acc",
            },
        }

    def load_embeddings(self, embeddings):
        """loads pretrained embeddings"""
        self.encoder.load_embeddings(embeddings)

    def on_save_checkpoint(self, checkpoint):
        """Modifies checkpoint before saving by removing word-embeddings"""
        del checkpoint["state_dict"]["encoder.embeddings.weight"]
        del checkpoint["state_dict"]["encoder.embeddings.bias"]

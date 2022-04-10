"""DataLoaders and helpers"""
import pytorch_lightning as pl


class SNLIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for SNLI dataset.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

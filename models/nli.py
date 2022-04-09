"""Generic NLI model"""
import pytorch_lightning as pl


class NLI(pl.LightningModule):
    """Generic NLI model"""

    def __init__(self):
        super().__init__()
        # TODO

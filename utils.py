import urllib.request

import torch
import pytorch_lightning as pl
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


class DownloadProgressBar(tqdm):
    """
    progress bar for downloads
    credit: https://stackoverflow.com/a/53877507/9889508
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class HackedLearningRateMonitor(pl.callbacks.LearningRateMonitor):
    """
    Custom Learning Rate monitor that allows to actually log the learning rate
    """

    def on_train_epoch_start(self, trainer, *args, **kwards):
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                for logger in trainer.loggers:
                    logger.log_metrics(
                        latest_stat,
                        step=trainer.fit_loop.epoch_loop._batches_that_stepped,
                    )
                    trainer.callback_metrics["lr_log"] = torch.Tensor(
                        [latest_stat["lr_log"]]
                    )


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

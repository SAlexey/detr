import pytorch_lightning as pl
import torch 


class MeanAveragePrecision(pl.metrics.Metric):

    def __init__(self) -> None:
        raise NotImplementedError()
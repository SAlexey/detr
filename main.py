from pathlib import Path
from datasets.data import MRIDataModule, MRISliceDataModule
import os
import time
from argparse import ArgumentParser
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
import torch

from models.model import MeDeCl
from util.callbacks import BestAndWorstCaseCallback, COCOEvaluationCallback, ModelMetricsAndLoggingBase


def get_argparse_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MeDeCl.add_argparse_args([parser])

    parser.add_argument("--experiment_name", default=f"Exp-{time.strftime('%d%m%YT%H:%M')}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--datadir", type=str, required=True)
    args = parser.parse_args()

    return args




@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig):
    args = get_argparse_args()
    
    model = instantiate(cfg.model)
    data = instantiate(cfg.data)
    logger = instantiate(cfg.logger) if "logger" in cfg else True
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks] if cfg.callbacks.extra else []       
    metrics_callback = ModelMetricsAndLoggingBase()
    callbacks = [checkpoint_callback, metrics_callback] + extra_callbacks
    trainer = pl.Trainer(**cfg.trainer, logger=logger)

    if resume_from_checkpoint:
        print(trainer.callbacks)
        print(resume_from_checkpoint)
        assert trainer.callbacks == callbacks
        trainer.callbacks = callbacks

    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    main()

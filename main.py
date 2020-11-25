import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from models.model import MeDeCl
from util.callbacks import MeDeClMetricsAndLogging


def get_argparse_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MeDeCl.add_argparse_args(parser)

    parser.add_argument("--experiment_name", default=f"Exp-{time.strftime('%d%M%YT%H:%m')}")
    args = parser.parse_args()

    return args

def main():
    args = get_argparse_args()
    
    model = MeDeCl(args)
    logger = pl.loggers.TensorBoardLogger(name=args.experiment_name)

    checkpoint_dirpath = os.path.join(args.weights_save_path, logger.name)

    if not os.path.exists(checkpoint_dirpath):
        os.mkdir(checkpoint_dirpath)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename="{epoch}_{validation_loss:.3f}"
    )

    metrics_callback = MeDeClMetricsAndLogging()

    trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=[checkpoint_callback]
        )

if __name__ == "__main__":
    main()

from pathlib import Path

from torchvision.models.resnet import resnet50
from datasets.data import MRIDataModule, MRISliceDataModule
import os
import time
from argparse import ArgumentParser

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

def main():
    args = get_argparse_args()
    
    model = MeDeCl(args)
    data = MRISliceDataModule(args)
    logger = pl.loggers.TensorBoardLogger(save_dir="tb_logs", name=args.experiment_name)

    checkpoint_root = Path(args.weights_save_path or "checkpoints")

    if not checkpoint_root.exists():
        checkpoint_root.mkdir()

    checkpoint_dirpath = checkpoint_root / logger.name

    if not checkpoint_dirpath.exists():
        checkpoint_dirpath.mkdir()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dirpath,
            filename="{epoch}_{validation_loss:.3f}",
            monitor="validation_loss",
        )

    resume_from_checkpoint = args.resume_from_checkpoint
    if args.resume:
        # resume training taking the last saved checkppoint
        checkpoints = sorted([ckpt for ckpt in checkpoint_dirpath.iterdir()])
        if checkpoints:
            resume_from_checkpoint = str(checkpoints[-1])
            
                

    metrics_callback = ModelMetricsAndLoggingBase()
    coco_eval_callback = COCOEvaluationCallback()
    best_and_worst_callback = BestAndWorstCaseCallback(5)
    callbacks = [checkpoint_callback, best_and_worst_callback, metrics_callback]
    trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=resume_from_checkpoint
        )

    if resume_from_checkpoint:
        assert trainer.callbacks == callbacks
        trainer.callbacks = callbacks

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()

from copy import copy
from datasets.coco import COCOWrapper
import io
import json
import os

from pycocotools.cocoeval import COCOeval
from util.box_ops import box_cxcywh_to_xyxy
from torchvision import ops

import torch
from datasets.coco_eval import CocoEvaluator
from typing import Callable, List
import pytorch_lightning as pl
from models.detr import PostProcess
from pycocotools.coco import COCO


class ModelMetricsAndLoggingBase(pl.callbacks.Callback):

    def __init__(
            self, 
            training_metrics:List[pl.metrics.Metric]=[], 
            validation_metrics:List[pl.metrics.Metric]=[], 
            test_metrics:List[pl.metrics.Metric]=[]
        ) -> None:
        self.training_metrics = training_metrics
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics

    def common_log(self, pl_module, outputs, batch, on_step=True, on_epoch=True, prefix=""):
        """
        Common Log for all training, validation and test steps
        Logs total loss as well as separate loss components
        Args:
            pl_module: LightningModule
            outputs: (dict) a dictionary containing following key, value pairs:
                loss_dict: (dict) a dictionary containing loss components
                loss: (float) total loss calculated form the loss_dict
                output: (dict) a dictionary containing model predictions
            batch: (tuple, n=2) a tuple with elements
                0: (tensor)  model inputs
                1: (tuple, n=batch_size) n dictionaries containing targets 
        KWargs:
            on_step: (bool, default=True) log on each step
            on_epoch: (bool, default=True) log reduced value on each epoch
            prefix: (string) a prefix prepended to each logged value e.g [training_, validation_, test_]
        """
        pl_module.log(f"{prefix}_loss", outputs["loss"], on_step=on_step, on_epoch=on_epoch)
        pl_module.log_dict({
            f"{prefix}_{k}": v for k, v in outputs["loss_dict"].items()
        }, on_step=on_step, on_epoch=on_epoch)

        for i, metric in enumerate(getattr(self, f"{prefix}_metrics", [])):
            metric_name = getattr(metric, "name", f"metric_{i}")
            metric(outputs["output"], batch)
            pl_module.log(f"{prefix}_{metric_name}", metric, on_step=metric.compute_on_step, on_epoch=on_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self.common_log(pl_module, outputs, batch, prefix="training")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self.common_log(pl_module, outputs, batch, prefix="validation")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self.common_log(pl_module, outputs, batch, prefix="test")


class COCOEvaluationCallback(pl.Callback):

    # this is a Hacker McHackface method to compute detector evaluation metrics
    def __init__(self, compute_frequency=10) -> None:
        self.frequency = compute_frequency
        self.postprocess = PostProcess()
        self.coco_evaluator:COCOeval = None
        self.coco_gt = None

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.current_epoch: return 
            
        if trainer.current_epoch == 1:
            # create coco_gt dataset
            coco_gt = COCO()
            dataset = pl_module.val_dataloader().dataset
            coco_gt.datatset = {
                "images": dataset.images, 
                "annotations": dataset.annotations, 
                "categories": dataset.categories
            }
            coco_gt.createIndex()
            self.coco_gt = coco_gt
            
        # reset coco evaluator 
        if trainer.current_epoch % self.frequency == 0:
            self.coco_evaluator = CocoEvaluator(self.coco_gt, ["bbox"])
            

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.coco_evaluator is not None and (trainer.current_epoch % self.frequency == 0):
            # run evaluation
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            torch.save(
                self.coco_evaluator, 
                f"coco_evaluator_verion_{trainer.logger.verion}_epoch_{trainer.current_epoch:03d}.pth"
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.coco_evaluator is not None and (trainer.current_epoch % self.frequency == 0):
            self.coco_evaluator.update(self.postprocess(outputs))


class ShowBestAndWorstCaseCallback(pl.Callback):

    def __init__(self, compute_frequency=10) -> None:
        self.frequency = compute_frequency


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.frequency == 0:
            image, targets = batch
            # image = tensor [batch_size, 1, 300, 300] in 2d and [batch_size, 160, 300, 300] in 3d
            # targets = tuple[dict{boxes, labels}, x batch_size]

            tgt_boxes = torch.stack([t["boxes"] for t in targets])
            tgt_labels = torch.stack([t["labels"] for t in targets])

            out_boxes = outputs["pred_boxes"]
            out_labels = outputs["pred_logits"].sigmoid(-1)[..., :-1]

            # convert both box sets to [x1, y1, x2, y2] format
            tgt_boxes = box_cxcywh_to_xyxy(tgt_boxes) * torch.FloatTensor((300,)*4)
            out_boxes = box_cxcywh_to_xyxy(out_boxes) * torch.FloatTensor((300,)*4)





            


class MeDeClMetricsAndLogging(ModelMetricsAndLoggingBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coco_evaluator = {
            "train": {},
            "val": {},
            "test": {}
        }
        self.postprocessors =  {'bbox': PostProcess()}

    def setup(self, trainer, pl_module, stage):
        if stage == "fit" or stage is None:
            coco = pl_module.val_dataloader().dataset.coco
            for key, coco_gt in coco.items():
                self.coco_evaluator["val"][key] = CocoEvaluator(coco_gt)

            coco = pl_module.train_dataloader().dataset.coco
            for key, coco_gt in coco.items():
                self.coco_evaluator["train"][key] = CocoEvaluator(coco_gt)



        if stage == "test" or stage is None:
            coco_dict = pl_module.test_dataloader().dataset.coco_dict
            for key, coco_gt in coco_dict.items():
                self.coco_evaluator["test"][key] = CocoEvaluator(coco_gt)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        inputs, targets = batch
        
        # box = [z1, x1, y1, z2, x2, y2]
        planes = (
            ("sag", [1, 2], [1, 2, 4, 5]),
            ("cor", [0, 1], [0, 1, 3, 4]), 
            ("axi", [0, 2], [0, 2, 3, 5])
        )
        
        for plane, ax, dims in planes:
            orig_target_sizes = torch.stack([t["orig_size"][ax] for t in targets], dim=0)
            _outputs = copy.deepcopy(outputs)
            # take the relevant components of bounding boxes
            _outputs["boxes"] = _outputs["boxes"][:, dims]
            results = self.postprocessors['bbox'](_outputs, orig_target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            self.coco_evaluator["val"][plane].update(res)

    def on_validation_epoch_end(self, trainer, pl_module):
        for key, coco_eval in self.coco_evaluator["val"].items():
            if (trainer.current_epoch % 50) == 0:
                coco_eval.accumulate()
                coco_eval.summarize()
                savepath = os.path.join(trainer.logger.dirpath, "coco_eval")
                torch.save(coco_eval, os.join(savepath, f"coco_{key}_{trainer.current_epoch}.eval"))
            
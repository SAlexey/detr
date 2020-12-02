from typing import Dict, Tuple

from pycocotools.cocoeval import COCOeval
from datasets.coco_eval import CocoEvaluator
import pytorch_lightning as pl
import torch
from torch import Tensor


class MeanAveragePrecision(pl.metrics.Metric):

    def __init__(
            self, 
            confidence_thresholds=torch.arange(0.2, 1, 0.1),
            iou_thresholds=torch.anrage(0.05, 0.95, 0.05)
        ) -> None:
        correct_labels = {f"{k}": 0 for k in confidence_thresholds}
        correct_boxes = {f"{}"}
        self.add_state("TP", {}, dist_reduce_fx="cat", persistent=False)


    def update(self, outputs:Dict[str, Tensor], targets:Tuple[Dict, ...]) -> None:
        
        out_classes = outputs["pred_logits"].softmax(-1)[..., :-1]
        out_boxes = outputs["pred_boxes"]

        tgt_classes = torch.cat([t["labels"] for t in targets])
        tgt_boxes = torch.cat([t["boxes"] for t in targets])



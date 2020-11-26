from typing import Tuple, Dict
import pytorch_lightning as pl
import torch
from torch import Tensor 


class MeanAveragePrecision(pl.metrics.Metric):

    def __init__(self) -> None:
        self.add_state("name", 1, dist_reduce_fx="cat", persistent=False)


    def update(self, outputs:Dict[str, Tensor], targets:Tuple[Dict, ...]) -> None:
        out_classes = outputs["pred_logits"].softmax(-1)[..., :-1]
        out_boxes = outputs["pred_boxes"]

        tgt_classes = torch.cat([t["labels"] for t in targets])
        tgt_boxes = torch.cat([t["boxes"] for t in targets])
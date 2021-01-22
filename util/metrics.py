from typing import Dict, Tuple
from pycocotools.coco import COCO

from pycocotools.cocoeval import COCOeval
from datasets.coco_eval import CocoEvaluator
import pytorch_lightning as pl
import torch
from torch import Tensor
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import average_precision_score
import numpy as np
from torchvision import ops



# class MeanAveragePrecision(pl.metrics.Metric):

#     def __init__(
#             self, 
#             setting="coco",
#             confidence_thresholds=torch.arange(0.2, 1, 0.1),
#             iou_thresholds=torch.anrage(0.05, 0.95, 0.05)
#         ) -> None:
#         correct_labels = {f"{k}": 0 for k in confidence_thresholds}
#         correct_boxes = {f"{}"}
#         self.add_state("TP", {}, dist_reduce_fx="cat", persistent=False)


#     def update(self, outputs:Dict[str, Tensor], targets:Tuple[Dict, ...]) -> None:
        
#         out_classes = outputs["pred_logits"].softmax(-1)[..., :-1]
#         out_boxes = outputs["pred_boxes"]

#         tgt_classes = torch.cat([t["labels"] for t in targets])
#         tgt_boxes = torch.cat([t["boxes"] for t in targets])


class DetectionAP(pl.metrics.Metric):
    """
    Average Precision for Object Detection
    
    Average precision is defined as the are under the precision-recall curve
    """

    def __init__(self, *args, iou_thd:float=0.5, pos_tgt:int=1, pos_weight:float=1.0, dim=2, **kwargs):
        """
        Initialize Metric
        
        Parameters: 
        -----------
        iou_thd: (float)
            the iou threshold used for computing TP/FP
            
        pos_tgt: (int)
            the label of the positive target
        
        pos_weight: (float)
            the weight of the positive target        
        """

        super().__init__(*args, **kwargs)
        self.iou_thd = iou_thd
        self.pos_tgt = pos_tgt
        
        self.dim = dim
        self.weight = pos_weight
        
        if self.dim != 2:
            raise NotImplementedError()
        
        self.add_state("out", default=[])
        self.add_state("tgt", default=[])
        
    def _iou(self, out_boxes, tgt_boxes):
        ious = ops.box_iou(out_boxes, tgt_boxes)
        return ious.diag()

    def update(self, out_boxes, out_probas, tgt_boxes, tgt_labels):
        """
        Updates Metric State
            
        Parameters:
        ----------
        out_boxes: (tensor[N, 4])
                a tensor containing bounding box predictions 
                in format [center_x, center_y, width, height]
                normalized by the image size
        
        out_probas: (tensor[N, num_classes])
                a tensor containing class probabilities
        
        tgt_boxes: (tensor[N, 4])
                a tensor containign bouding box targets 
                same format and shape as out_boxes
                
        tgt_labels: (tensor[N, 1])
                a tensor containing target classes ids
        
        Example:
        
        avg_precision = DetectionAP(pos_tgt=1)  # <- average precision for class 1 
        ...
        inputs, targets = batch
        out = model(inputs)
        avg_precision(out["boxes"], out["probas"], targets["boxes"], targets["labels"])  # <- update the state
        """

        
        
        iou = self._iou(out_boxes, tgt_boxes)   # compute iou values between target and output boxes
        
        iou = (iou >= self.iou_thd)             # binarize the iou vector
        
        tgt = (tgt_labels == self.pos_tgt)    # binarize the target vector in one-vs-rest fashion  
        tgt = tgt & iou                         # mask target with the iou values

        self.out.append(out_probas.float()) 
        self.tgt.append(tgt.long()) 
        
    def compute(self):
        
        """
        Computes Average Precision         
        
        Example: 
        avg_precision = DetectionAP(pos_tgt=1)  # <- average precision for class 1 
        ...
        for batch in loader:
            out_boxes, out_probas = model(batch)
            avg_precision(out_boxes, out_probas, batch["tgt_boxes"], batch["tgt_labels"]) # <- updates the state
        value = avg_precision.compute() # <- compute avg precision from the accumulated state
        """
        
        out = torch.cat(self.out).flatten().cpu().numpy()
        tgt = torch.cat(self.tgt).flatten().cpu().numpy()

        sample_weight = None

        if self.weight is not None:
            sample_weight = tgt * self.weight
        
        average_precision = average_precision_score(tgt, out, sample_weight=sample_weight)
        return average_precision

        
        # iou_bin = (iou > self.iou_thd).astype(int)
        
        # tgt = tgt * iou_bin  
        
        # weight = np.array([self.weight.get(cl, 1.0) for cl in tgt], dtype=float)
        
        # desc_score_indices = np.argsort(tgt, kind="mergesort")[::-1]

        # y_score = out[desc_score_indices]
        # y_true = tgt[desc_score_indices]
        
        # distinct_value_indices = np.where(np.diff(y_score))[0]
        # threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        
            
        # iou = iou[desc_score_indices]
        # out = out[desc_score_indices]
        # tgt = tgt[desc_score_indices]
        
        # tps = stable_cumsum(pos_bin * weight)[threshold_idxs]
        # fps = stable_cumsum((1 - pos_bin) * weight)[threshold_idxs]
        
        # precision = tps / (tps + fps)
        # precision[np.isnan(precision)] = 0
        # recall = tps / tps[-1]

        # last_ind = tps.searchsorted(tps[-1])
        # sl = slice(last_ind, None, -1)
        
        # precision, recall = np.r_[precision[sl], 1], np.r_[recall[sl], 0]
        # bin_ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        
        

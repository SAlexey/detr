import os
import tempfile
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Callable, List

import numpy as np

import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from datasets.coco import COCOWrapper
from datasets.coco_eval import CocoEvaluator
from models.detr import PostProcess
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from tqdm import tqdm

from util.box_ops import BboxFormatter, box_cxcywh_to_xyxy
from util.gradcam import GradCam

COCO_EVAL_NAMES = ("AP_coco", "AP_pascal", "AP_strict", "AP_small", "AP_medium", "AP_large", "AR_max_1", "AR_max_10", "AR_max_100", "AR_small", "AR_medium", "AR_large")
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
        Logs totalModelMetricsAndLoggingBaseloss as well as separate loss components
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


def coco_gt_from_dataset(loader, device="cpu"):
    coco_api = COCO()
            
    images = []
    annotations = []
    categories = []
    category_ids = set()

    box_convert = loader.dataset.formatter.convert
    pbar = tqdm(loader, total=len(loader.dataset))
    for item in pbar:
        _, targets = item
        # create coco_gt
        for target in targets:
            images.append({
                "id": target["image_id"],
                "width": target["orig_size"][0],
                "height": target["orig_size"][1],
            })
            
            for box, label in zip(target["boxes"], target["labels"]):
                label = label.item()
                box = box_convert(box, "ccwh", "xywh").to(device)
                box = box * torch.tensor(target["orig_size"] * 2, device=device, dtype=torch.float32)
                annotations.append({
                    "id": len(annotations),
                    "image_id": target["image_id"],
                    "category_id": label,
                    "iscrowd": 0,
                    "area": box[2] * box[3],
                    "bbox": list(box) # box = [x, y, width, height]
                })
                if label not in category_ids:
                    categories.append({"id": label, "name": f"meniscus_{label}"})
                    category_ids.add(label)
            pbar.update(1)
    coco_api.dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations
    }

    coco_api.createIndex()
    return coco_api


class COCOEvaluationCallback(pl.Callback):

    def __init__(self, compute_frequency=10):
        self.frequency = compute_frequency

        self.coco_dts = defaultdict(lambda: {"images": [], "annotations": [], "categories": []})
        self.coco_gts = defaultdict(COCO)

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 1 or ( trainer.current_epoch % self.frequency == 0 ):
            self.coco_eval = CocoEvaluator(self.coco_gt, ["bbox"])
    
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # create coco_gt
        if trainer.current_epoch == 1 or ( trainer.current_epoch % self.frequency == 0):
            for key, coco_eval in self.coco_eval.items():
                coco_eval.synchronize_between_processes()
                coco_eval.accumulate()
                coco_eval.summarize()

                if trainer.logger is not None:
                    save_dir = os.path.join(str(trainer.logger.save_dir), str(trainer.logger.name), f"version_{trainer.logger.version}")
                    torch.save(
                        coco_eval.coco_eval["bbox"], 
                        os.path.join(save_dir, f"coco_evaluator_{key}_epoch_{trainer.current_epoch:03d}.pth")
                    )
                    
                    trainer.logger.log_metrics({
                        f"validation_{key}_{name}": value
                        for name, value in zip(
                        COCO_EVAL_NAMES,
                        coco_eval.coco_eval['bbox'].stats.tolist())
                    }, step=trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):     
        if self.current_epoch == 0:
            pass



        _, targets = batch
        out = outputs["output"]
        if trainer.current_epoch == 1 or (trainer.current_epoch % self.frequency == 0):
            for _, coco_eval in self.coco_eval.items():
                orig_target_sizes = torch.stack([
                    torch.tensor(tuple(t["orig_size"]), dtype=torch.int16) 
                    for t in targets
                ])
                targets = [{k: v for k, v in t.items()} for t in targets]
                results = self.postprocess(out, orig_target_sizes)
                res = {target['image_id']: output for target, output in zip(targets, results)}
                coco_eval.update(res)


class BestAndWorstCaseCallback(pl.Callback):

    def __init__(self, compute_frequency=10) -> None:
        self.frequency = compute_frequency


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.frequency == 0:
            convert = pl_module.val_dataloader().dataset.formatter.convert
            images, targets = batch
            # image = tensor [batch_size, 1, 300, 300] in 2d and [batch_size, 160, 300, 300] in 3d
            # targets = tuple[dict{boxes, labels}, x batch_size]

            tgt_boxes = torch.cat([t["boxes"] for t in targets])
            tgt_labels = torch.cat([t["labels"] for t in targets])

            out_boxes = outputs["output"]["pred_boxes"].flatten(0, 1)
            out_labels = outputs["output"]["pred_logits"].flatten(0, 1).softmax(-1)

            # convert both box sets to [x1, y1, x2, y2] format
            tgt_box_xyxy = convert(tgt_boxes, "ccwh", "xyxy").to(pl_module.device)  
            tgt_boxes = tgt_box_xyxy * torch.tensor((300,)*4, device=pl_module.device, dtype=torch.float32)
            out_boxes = convert(out_boxes, "ccwh", "xyxy") * torch.tensor((300,)*4, device=pl_module.device, dtype=torch.float32)
            ious = torch.diag(box_iou(out_boxes, tgt_boxes))

            fig, axes = plt.subplots(ncols=2)

            for ax, f, title in zip(axes, (torch.max, torch.min), ("best", "worst")):

                case = f(ious, -1)
                img = images.tensors[case.indices]
                obox = out_boxes[case.indices].squeeze()
                tbox = tgt_boxes[case.indices].squeeze()

                ax.imshow(img.squeeze(), "gray")
                ax.add_patch(plt.Rectangle((tbox[1], tbox[0]), tbox[3] - tbox[1], tbox[2] - tbox[0],  fill=False, ec="blue"))
                ax.add_patch(plt.Rectangle((obox[1], obox[0]), obox[3] - obox[1],  obox[2] - obox[0], fill=False, ec="green" if case.values > 0.5 else "red"))

                ax.set_title(f"{title} case".title())

            trainer.logger.experiment.add_figure(f"Epoch_{trainer.current_epoch}_extremes.png", fig)


class VisualizeLayerCallback(pl.Callback):
    def __init__(self, frequency=10):
        self.frequency = frequency


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
        
class GradCamCallback(VisualizeLayerCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_cam = None
        self.cams = {
            "training": [],
            "validation": []
        }

    def on_fit_start(self, trainer, pl_module):
        self.grad_cam = GradCam(model=pl_module.model, feature_module=pl_module.model.layer4, target_layer_names=["2"])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if trainer.current_epoch % self.frequency == 0 and (10 <= batch_idx <= 40) and (batch_idx % 2 == 0):
            self.visualize_grad_cam(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if trainer.current_epoch % self.frequency == 0 and (10 <= batch_idx <= 40) and (batch_idx % 2 == 0):
            self.visualize_grad_cam(trainer, pl_module, outputs[0][0]["extra"], batch, batch_idx)

    def visualize_grad_cam(self, trainer, pl_module, outputs, batch, batch_idx):      
        with torch.set_grad_enabled(True):
            inputs, targets = batch
            cams = [self.grad_cam(img.unsqueeze(0).to(pl_module.device)) for img in inputs] # get class activations mappings
            cams = [np.stack(cam) for cam in cams] # make them 3 chanels

            fig, axes = plt.subplots(ncols=2, figsize=(10, 6))
            predictions = outputs["output"].softmax(-1).squeeze()
            predictions = predictions.max(-1)

            scores = predictions.values

            best_score = scores.max(-1)
            worst_score = scores.max(-1)

            scores = [best_score, worst_score]

            labels = [predictions.indices[each.indices] for each in scores]
            
            for ax, img, cam, score, label, target  in zip(axes, inputs, cams, scores, labels, targets):
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + np.moveaxis(img.numpy(), 0, -1)
                cam = cam / np.max(cam)
                ax.imshow(cam)
                ax.set_title(f"Is: {target}; got: {label} ({score.values * 100:.1f}%)")
                plt.axis('off')
            prefix = "training" if pl_module.training else "validation"
            trainer.logger.experiment.add_figure(f"GradCam_{prefix}_epoch_{trainer.current_epoch}_{batch_idx}", fig)


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
            

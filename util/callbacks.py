import os
import tempfile
from collections import defaultdict
from copy import copy
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Callable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from datasets.coco_eval import CocoEvaluator
from models.detr import PostProcess
from pycocotools.coco import COCO
from torch import nn
from torchvision import ops
from torchvision.ops import box_iou
from tqdm import tqdm

from util.gradcam import GradCam
from util.metrics import DetectionAP
from mean_average_precision import MeanAveragePrecision

STAGE = Enum("STAGE", (("TRAIN", "training"), ("VAL", "validation"), ("TEST", "test")))

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

# class COCOEvaluationCallback(pl.Callback):

#     def __init__(self, compute_frequency=10):
#         self.frequency = compute_frequency

#         self.coco_dts = defaultdict(lambda: {"images": [], "annotations": [], "categories": []})
#         self.coco_gts = defaultdict(COCO)

#     def on_validation_epoch_start(self, trainer, pl_module):
#         if trainer.current_epoch == 1 or ( trainer.current_epoch % self.frequency == 0 ):
#             self.coco_eval = CocoEvaluator(self.coco_gt, ["bbox"])
    
    
#     def on_validation_epoch_end(self, trainer, pl_module):

#         # create coco_gt
#         if trainer.current_epoch == 1 or ( trainer.current_epoch % self.frequency == 0):
#             for key, coco_eval in self.coco_eval.items():
#                 coco_eval.synchronize_between_processes()
#                 coco_eval.accumulate()
#                 coco_eval.summarize()

#                 if trainer.logger is not None:
#                     save_dir = os.path.join(str(trainer.logger.save_dir), str(trainer.logger.name), f"version_{trainer.logger.version}")
#                     torch.save(
#                         coco_eval.coco_eval["bbox"], 
#                         os.path.join(save_dir, f"coco_evaluator_{key}_epoch_{trainer.current_epoch:03d}.pth")
#                     )
                    
#                     trainer.logger.log_metrics({
#                         f"validation_{key}_{name}": value
#                         for name, value in zip(
#                         COCO_EVAL_NAMES,
#                         coco_eval.coco_eval['bbox'].stats.tolist())
#                     }, step=trainer.current_epoch)

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):     
#         if self.current_epoch == 0:
#             pass

#         _, targets = batch
#         out = outputs["output"]
#         if trainer.current_epoch == 1 or (trainer.current_epoch % self.frequency == 0):
#             for _, coco_eval in self.coco_eval.items():
#                 orig_target_sizes = torch.stack([
#                     torch.tensor(tuple(t["orig_size"]), dtype=torch.int16) 
#                     for t in targets
#                 ])
#                 targets = [{k: v for k, v in t.items()} for t in targets]
#                 results = self.postprocess(out, orig_target_sizes)
#                 res = {target['image_id']: output for target, output in zip(targets, results)}
#                 coco_eval.update(res)


class EvaluateObjectDetection(pl.callbacks.Callback):

    def __init__(self, on_validation=True, on_training=False, on_test=False, setting="coco", num_classes=1,):
        self.on_validation = on_validation
        self.on_training = on_training
        self.on_test = on_test 
        self.setting = setting
        self._metric = MeanAveragePrecision(num_classes)    
        self._result = None
    

    def _should_compute(self, stage):
        return getattr(self, f"on_{stage.value}")

    
    def _evaluate_object_detection(self, stage, trainer, pl_module):

        if not self._should_compute(stage): return

        
        self._result = self._metric.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))
        pass
    
    def _add_features(self, stage, trainer, pl_module, outputs, batch, *args, **kwargs):
        if not self._should_compute(stage): return
        
       
        _, targets = batch 


        indices = pl_module.criterion.matcher(outputs, targets)
        idx = pl_module.criterion._get_src_permutation_idx(indices)

        out_bboxes = outputs["pred_boxes"][idx]
        out_probas = outputs["pred_logits"][idx]

        out_probas = out_probas.softmax(-1)   # [batch_size * num_queries, num_classes]

        tgt_bboxes = torch.cat([t["boxes"] for t in targets])
        tgt_labels = torch.cat([t["labels"] for t in targets])

        difficult = torch.zeros_like(tgt_labels)
        crowd = torch.zeros_like(difficult)
        
        out_bboxes = ops.box_convert(out_bboxes, "cxcywh", "xyxy")
        tgt_bboxes = ops.box_convert(tgt_bboxes, "cxcywh", "xyxy")

        tgt = torch.stack([tgt_labels, difficult, crowd], -1)
        tgt = torch.cat([tgt_bboxes, tgt], -1).cpu().numpy()

        preds = out_probas.max(-1)
        preds = torch.stack([preds.indices, preds.values], -1)
        preds = torch.cat([tgt_bboxes, preds], -1).cpu().numpy()

        self._metric.add(preds, tgt)


    def on_train_batch_end(self, *args, **kwargs):
        self._add_features(STAGE.TRAIN, *args, **kwargs)

    def on_validation_batch_end(self, *args, **kwargs):
        self._add_features(STAGE.VAL, *args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self._add_features(STAGE.TEST, *args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        self._evaluate_object_detection(STAGE.TRAIN, *args, **kwargs)
    
    def on_validation_epoch_end(self, *args, **kwargs):
        self._evaluate_object_detection(STAGE.VAL, *args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self._evaluate_object_detection(STAGE.TEST, *args, **kwargs)
    
    

    


# class ImageLoggingMixin(object):

#     def _format_image_name(self, module, stage, name="image_results"):
#         return f"{stage.value}_epoch={module.current_epoch}_step={module.global_step}_{name}"

#     def _log_image(self, pl_module:pl.LightningModule, stage, name, image):
#         name = self._format_image_name(pl_module, stage, name)
#         try:
#             pl_module.logger.experiment.add_figure(name, image)
#         except:
#             pass

#     def _prep_image(self, n):
#         if n < 4:
#             return plt.subplots(ncols=n, figsize=(10, 4))
#         else: 
#             ncols = 4 
#             nrows = ceil(n/ncols)
#             figsize = (10, 10//4*nrows)
#             return plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)



# class VisualizeAttentionOnImage(pl.Callback, ImageLoggingMixin):

#     def __init__(self, 
#         frequency=1, 
#         n_images=1, 
#         selector=None, 
#         on_training=False, 
#         on_validation=True, 
#         on_test=True, 
#         feature_layer_getter=None, 
#         encoder_layer_getter=None, 
#         decoder_layer_getter=None, 
#         dowsnsampling_factor=32
#     ):
#         self.f = frequency                  # step frequency 1 step = 1 batch
#         self.n = n_images                   # n <= batch_size!!
        
#         self.selector = selector            # samples = selector(output, target)
#         self.on_training = on_training      # compute for training batches?
#         self.on_validation = on_validation  # compute for validation batches?
#         self.on_test = on_test

#         self.get_encoder_layer = encoder_layer_getter
#         self.get_decoder_layer = decoder_layer_getter
#         self.get_feature_layer = feature_layer_getter  

#         self.fct = dowsnsampling_factor

#         self._conv_features = torch.empty(0)
#         self._enc_attn_weights = torch.empty(0)
#         self._dec_attn_weights = torch.empty(0)

#         self._validate_init()

    
#     def _validate_init(self):
#         assert callable(self.get_encoder_layer)
#         assert callable(self.get_encoder_layer)
#         assert callable(self.get_feature_layer)

#         if self.selector is not None:
#             assert callable(self.selector)
    

#     def _reset_state(self):
#         for hook in self._registered_hooks: hook.remove()
        
#         self._conv_features = torch.empty(0)
#         self._enc_attn_weights = torch.empty(0)
#         self._dec_attn_weights = torch.empty(0)
#         self._registered_hooks = []

#     def _add_hooks(self, module):
#         feature_layer:nn.Module = self.get_feature_layer(module)
#         encoder_layer:nn.Module = self.get_encoder_layer(module)
#         decoder_layer:nn.Module = self.get_decoder_layer(module)

#         feature_hook = feature_layer.register_forward_hook(
#             lambda it, ins, out: self._state.update({"conf_features": out})
#         )

#         encoder_hook = encoder_layer.register_forward_hook(
#             lambda it, ins, out: self._state.update({"enc_attn_weights": out[1]})
#         )

#         decoder_hook = decoder_layer.register_forward_hook(
#             lambda it, ins, out: self._state.update({"dec_attn_weights": out[1]})
#         )

#         self._registered_hooks.extend([feature_hook, encoder_hook, decoder_hook])

#     def _should_compute(self, stage, global_step):
#         return getattr(self, f"on_{stage.value}") and global_step % self.f == 0

#     def _compute_features(self):
#         shape = self.conv_features.shape[-2:]
#         self.enc_attn_weights = self.enc_attn_weights[0]
        
#         return self.dec_attn_weights, 

#     def _get_bbox_reference_points(self, bbox):
#         pass


#     def _visualize_attention(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, *args, **kwargs):

#         inputs, targets = batch
#         matcher = pl_module.criterion.matcher 
#         indices = matcher(outputs, targets)
        
        
#         h, w = self.conv_features['0'].tensors.shape[-2:]
#         dec_attn = self.dec_attn_weights[0, idx].view(h, w)

#         fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
#         colors = COLORS * 100
#         for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
#             ax = ax_i[0]
#             ax.imshow()
#             ax.axis('off')
#             ax.set_title(f'query id: {idx.item()}')
#             ax = ax_i[1]
#             ax.imshow(im)
#             ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                     fill=False, color='blue', linewidth=3))
#             ax.axis('off')
#             ax.set_title(CLASSES[probas[idx].argmax()])
#         fig.tight_layout()
  
#     def setup(self, trainer, pl_module, stage: str):

#         assert isinstance(nn.Module, self.get_feature_layer(pl_module))
#         assert isinstance(nn.Module, self.get_decoder_layer(pl_module))
#         assert isinstance(nn.Module, self.get_encoder_layer(pl_module))


#     def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
#         if self._should_compute(STAGE.TRAIN, pl_module.global_step): self._add_hooks()
        

#     def on_validation_batch_start(self, _, pl_module, *args, **kwargs):
#         if self._should_compute(STAGE.VAL, pl_module.global_step): self._add_hooks()

#     def on_test_batch_start(self, _, pl_module, *args, **kwargs):
#         if self._should_compute(STAGE.TEST, pl_module.global_step): self._add_hooks()

#     def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
#         if self._should_compute(STAGE.TRAIN, pl_module.global_step): 
#             self._visualize_attention(trainer, pl_module, *args, **kwargs)
#             self._reset_state()

#     def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
#         if self._should_compute(STAGE.VAL, pl_module.global_step): 
#             self._visualize_attention(trainer, pl_module, *args, **kwargs)
#             self._reset_state()

#     def on_test_batch_end(self, trainer, pl_module, *args, **kwargs):
#         if self._should_compute(STAGE.TEST, pl_module.global_step): 
#             self._visualize_attention(trainer, pl_module, *args, **kwargs)
#             self._reset_state()


# class VisualizeBBoxOnImage(pl.Callback):

#     def __init__(self, frequency=1, n_images=1, selector=None, on_training=False, on_validation=True, on_test=True):
#         self.f = frequency                  # step frequency 1 step = 1 batch
#         self.n = n_images                   # n <= batch_size!!
        
#         if selector is not None:
#             assert callable(selector)

#         self.selector = selector            # samples = selector(output, target)
#         self.on_training = on_training      # compute for training batches?
#         self.on_validation = on_validation  # compute for validation batches?
#         self.on_test = on_test
 

#     def _log_image(self, pl_module:pl.LightningModule, image, prefix=""):
#         image_name = prefix + f"epoch={pl_module.current_epoch}_step={pl_module.global_step}_bboxes"
#         try:
#             pl_module.logger.experiment.add_figure(image_name, image)
#         except: 
#             pass

#     def _prep_image(self, n):
#         if n < 4:
#             return plt.subplots(ncols=n, figsize=(10, 4))
#         else: 
#             ncols = 4 
#             nrows = ceil(n/ncols)
#             figsize = (10, 10//4*nrows)
#             return plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

#     def _visualize_bbox_on_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage):
#         inputs, targets = batch
#         if self.selector is not None:
#             inputs, outputs, targets = self.selector(inputs, outputs, targets)
#         else:
#             inputs = inputs[:self.n]            
#             outputs = outputs[:self.n]
#             targets = targets[:self.n]

#         inputs = inputs.cpu()

#         image, plots = self._prep_image(len(targets))

#         (*_, ih, iw) = inputs.shape
#         scale =  torch.as_tensor((ih, iw, ih, iw))

#         indices = self.criterion.matcher(outputs, targets)
#         idx = self.criterion._get_src_permutation_idx(indices)
        
#         src_boxes = outputs['pred_boxes'][idx]
#         tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#         ious = ops.box_iou(ops.box_convert(src_boxes, "cxcywh", "xyxy"), ops.box_convert(tgt_boxes, "cxcywh", "xyxy")).diag()

#         for ax, im, out_bb, tgt_bb, iou in zip(plots, inputs, src_boxes, tgt_boxes, ious):

#             ax.imshow(im[0], "gray")

#             out_bb = out_bb.cpu()
#             tgt_bb = tgt_bb.cpu()
            
#             tgt_boxes_scaled = tgt_bb * scale 
#             out_boxes_scaled = out_bb * scale
                        
#             out_xywh = ops.box_convert(out_boxes_scaled, "cxcywh", "xywh")
#             tgt_xywh = ops.box_convert(tgt_boxes_scaled, "cxcywh", "xywh")


#             # print(tgt_xywh)
#             x, y, w, h = tgt_xywh.unbind()
#             tgt_rect = plt.Rectangle((y, x), h, w, fill=False, ec="darkgreen")

#             x, y, w, h = out_xywh.unbind()
#             out_rect = plt.Rectangle((y, x), h, w, fill=False, ec="orange")
            
#             ax.add_patch(tgt_rect)
#             ax.add_patch(out_rect)

#             ax.text(y - 10, x, f"iou={iou:.2f}", fontsize=10, bbox=dict(facecolor='darkgreen', alpha=0.5))
        
#         self._log_image(pl_module, image, prefix=stage.value)


#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         if self.on_training and pl_module.global_step % self.f == 0: 
#             self._visualize_bbox_on_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, STAGE.TRAIN)

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         if self.on_validation and pl_module.global_step % self.f == 0: 
#             self._visualize_bbox_on_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, STAGE.VAL)

#     def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         if self.on_test and pl_module.global_step% self.f == 0:
#             self._visualize_bbox_on_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, STAGE.TEST)

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
            

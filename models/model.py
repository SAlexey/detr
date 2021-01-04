from argparse import ArgumentParser, Namespace
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from hubconf import detr_resnet101_dc5
from pickle import NONE
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from numpy.lib.arraysetops import isin
from torch import nn
from torchvision.models.resnet import resnet50

from models.backbone import build_backbone
from models.detr import DETR, PostProcess, SetCriterion, build
from models.matcher import build_matcher
from models.transformer import build_transformer
from omegaconf import DictConfig
from hydra.utils import instantiate
import torchio as tio

class LitModel(pl.LightningModule):

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, hparams:Optional[Namespace]=None) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters(hparams)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr": self.hparams.lr_transformer
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def common_step(self, batch):
        inputs = batch["image"][tio.DATA].squeeze(1).unbind(0)
        inputs = nested_tensor_from_tensor_list(list(inputs))
        output = self.forward(inputs)
        losses_dict = self.criterion(output, batch)
        weight_dict = self.criterion.weight_dict
        output["loss_dict"] = losses_dict
        output["loss"] = sum(weight_dict[k] * losses_dict[k] for k in weight_dict)
        return output
    
    def training_step(self, batch, *args, **kwargs):
        out = self.common_step(batch)
        loss_dict = {f"training_{loss}": value for loss, value in out["loss_dict"].items()}
        self.log_dict(loss_dict, on_epoch=True, on_step=False)
        return out

    def validation_step(self, batch, *args, **kwargs):
        out = self.common_step(batch)
        loss_dict = {f"validation_{loss}": value for loss, value in out["loss_dict"].items()}
        self.log_dict(loss_dict, on_epoch=True, on_step=False)
        return out

    def test_step(self, batch, *args, **kwargs):
        return self.common_step(batch)


class LitBackbone(pl.LightningModule):

    def __init__(self, hparams):
        self.model = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        self.criterion = nn.CrossEntropyLoss()
        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, conv1.out_channels, conv1.kernel_size, conv1.stride, conv1.padding, bias=False)
        self.model.conv1.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)
        self.model.fc = nn.Linear(2048, 3, bias=True)
        self.accuracy = pl.metrics.classification.Accuracy()

    def common_step(self, batch):
        inputs, targets = batch
        output = self.forward(inputs.permute((1, 0, 2, 3)))
        loss = self.criterion(output, targets.squeeze())
        return {"output": output, "loss": loss}

    def training_step(self, batch, *args, **kwargs):
        out = super().training_step(batch, *args, **kwargs)
        predictions = out["output"].softmax(-1).max(-1, keepdim=True)
        self.accuracy(predictions.indices.squeeze(), batch[1].squeeze())
        self.log("training_accuracy", self.accuracy, on_epoch=True)
        return out

    def validation_step(self, batch, *args, **kwargs):
        out = super().validation_step(batch, *args, **kwargs)
        predictions = out["output"].softmax(-1).max(-1, keepdim=True)
        self.accuracy(predictions.indices.squeeze(), batch[1].squeeze())
        self.log("validation_accuracy", self.accuracy, on_epoch=True)
        return out

# class MeDeCl(Detector):
#     def __init__(self, args) -> None:

#         backbone = build_backbone(args)


#         if args.backbone_checkpoint_path:
#             checkpoint = torch.load(args.backbone_checkpoint_path)
#             state_dict = checkpoint.get("state_dict")
#             backbone.load_state_dict({k.replace("model.", "0.body."): v for k,v in state_dict.items() if "fc." not in k})
#             print("backbone checkpoint loaded")

#         transformer = build_transformer(args)

#         matcher = build_matcher(args)

#         criterion = SetCriterion(
#             args.num_classes, matcher, {
#                 "loss_ce": 1,
#                 "loss_bbox": args.bbox_loss_coef,
#                 "loss_giou": args.giou_loss_coef,
#             }, eos_coef=args.eos_coef,
#             losses = ['labels', 'boxes']
#         )

#         input_proj = (torch.nn.Conv2d if args.input_dim == "2d" else torch.nn.Conv3d)(
#             backbone.num_channels, transformer.d_model, kernel_size=1
#         )

#         model = DETR(
#             backbone, 
#             input_proj,
#             transformer, 
#             args.num_classes, 
#             args.num_queries, 
#             aux_loss=args.aux_loss
#         )

#         super().__init__(model, criterion, args)

#     def configure_optimizers(self):
#         return torch.optim.AdamW([
#             {"params": self.model.backbone.parameters(), "lr": self.hparams.lr_backbone},
#             {"params": self.model.transformer.parameters(), "lr": self.hparams.lr_transformer}
#         ], self.hparams.lr, weight_decay=self.hparams.weight_decay)

# class DETR101_DC5(Detector):

#     def __init__(self, cfg:DictConfig):
#         model, postprocessors = detr_resnet101_dc5(pretrained=True, return_postprocessor=True)
#         self.postprocessors = {"bbox": postprocessors}
#         body = model.backbone[0].body
#         conv1_weight = body.conv1.weight.data.mean(dim=1, keepdim=True)
#         body.conv1 = nn.Conv2d(1, body.conv1.out_channels, kernel_size=7, stride=2, padding=3,
#                             bias=False)
#         body.conv1.weight.data = conv1_weight
        
#         matcher = instantiate(cfg.matcher)
#         criterion = instantiate(cfg.criterion, matcher)
#         super().__init__(model, criterion, postprocessors=postprocessors)

# class DetrMRI(Detector):

#     def __init__(self, args):
#         model, postprocessors = detr_resnet101_dc5(pretrained=True, return_postprocessor=True)
#         self.postprocessors = {"bbox": postprocessors}
#         body = model.backbone[0].body
#         conv1_weight = body.conv1.weight.data.mean(dim=1, keepdim=True)
#         body.conv1 = nn.Conv2d(1, body.conv1.out_channels, kernel_size=7, stride=2, padding=3,
#                             bias=False)
#         body.conv1.weight.data = conv1_weight
#         matcher = build_matcher(args)

#         criterion = SetCriterion(
#             args.num_classes, matcher, {
#                 "loss_ce": 1,
#                 "loss_bbox": args.bbox_loss_coef,
#                 "loss_giou": args.giou_loss_coef,
#             }, eos_coef=args.eos_coef,
#             losses = ['labels', 'boxes']
#         )
#         super().__init__(model, criterion, args)


#     def configure_optimizers(self):
#         param_dicts = [
#             {
#                 "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
#                 "lr": self.hparams.lr_transformer
#             },
#             {
#                 "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
#                 "lr": self.hparams.lr_backbone,
#             },
#         ]
#         return torch.optim.AdamW(param_dicts, self.hparams.lr, weight_decay=self.hparams.weight_decay)









    


        



    
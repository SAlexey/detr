from argparse import Namespace
from enum import Enum
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models.resnet import resnet50
from util.misc import nested_tensor_from_tensor_list

STAGE = Enum(
    "STAGE", [("TRAIN", "training"), ("VAL", "validation"), ("TEST", "testing")]
)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        hparams: Optional[Namespace] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters(hparams)

    def forward(self, batch):
        inputs = self._format_inputs(batch)
        output = self.model(inputs)
        return output

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": self.hparams.lr_transformer,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.lr_backbone,
            },
        ]
        return torch.optim.AdamW(
            param_dicts, self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    @torch.no_grad()
    def _format_inputs(self, inputs):
        return nested_tensor_from_tensor_list(list(inputs.unbind(0)))

    def _compute_loss(self, output, target, stage=STAGE.TRAIN):
        losses_dict = self.criterion(output, target)
        weight_dict = self.criterion.weight_dict
        loss = sum(weight_dict[k] * losses_dict[k] for k in weight_dict)
        losses_dict = {
            f"{stage.value}_{loss}": value for loss, value in losses_dict.items()
        }
        return loss, losses_dict

    def training_step(self, batch, *args, **kwargs):
        inputs, target = batch
        output = self.forward(inputs)
        loss, loss_dict = self._compute_loss(output, target, stage=STAGE.TRAIN)
        self.log("training_loss", loss, on_epoch=True, on_step=False)
        self.log_dict(loss_dict, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        inputs, target = batch
        output = self.forward(inputs)
        loss, loss_dict = self._compute_loss(output, target, stage=STAGE.VAL)
        self.log("validation_loss", loss)
        self.log_dict(loss_dict, on_epoch=True, on_step=False)
        return output


class LitBackbone(pl.LightningModule):
    def __init__(self, hparams):
        self.model = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, True]
        )
        self.criterion = nn.CrossEntropyLoss()
        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            1,
            conv1.out_channels,
            conv1.kernel_size,
            conv1.stride,
            conv1.padding,
            bias=False,
        )
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

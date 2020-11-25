from typing import Any, Callable, List, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
import torch
from torch import ones
from torch.nn.modules.module import T_co 


class ModelBase(pl.LightningModule):

    def __init__(self, model, criterion) -> None:
        super.__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return self.model(*input)

    def reduce_loss(self, loss_dict):
        weight = getattr(self.criterion, "weight_dict", {})
        return sum((val * weight.get(key, 1) for key, val in loss_dict.items()))

    def common_step(self, batch):
        inputs, targets = batch
        output = self.forward(inputs)
        loss_dict = self.criterion(output, targets)
        loss = self.reduce_loss(loss_dict)
        return {"output": output, "loss_dict": loss_dict, "loss": loss}

    def training_step(self, batch, *args, **kwargs):
        return self.common_step(batch)    

    def validation_step(self, batch, *args, **kwargs):
        return self.common_step(batch)

    def test_step(self, batch, *args, **kwargs):
        return self.common_step(batch)


class ModelMetricsAndLogging(pl.Callback):

    def __init__(
            self, 
            training_metrics:List[Callable]=[], 
            validation_metrics:List[Callable]=[], 
            test_metrics:List[Callable]=[]
        ) -> None:
        self.training_metrics = training_metrics
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics

    def common_log(self, pl_module, outputs, batch, on_step=True, on_epoch=True, prefix=""):
        pl_module.log(f"{prefix}_loss", outputs["loss"], on_step=on_step, on_epoch=on_epoch)
        pl_module.log_dict({
            f"{prefix}_{k}": v for k, v in outputs["loss_dict"].items()
        }, on_step=on_step, on_epoch=on_epoch)

        for i, metric in enumerate(getattr(self, f"{prefix}_metrics", [])):
            metric_name = getattr(metric, "name", f"{prefix}_metric_{i}")
            metric(outputs["output"], batch)
            pl_module.log(metric_name, metric, on_step=on_step, on_epoch=on_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self.common_log(pl_module, outputs, batch, prefix="training")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self.common_log(pl_module, outputs, batch, prefix="validation")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        self.common_log(pl_module, outputs, batch, prefix="test")


    


        



    
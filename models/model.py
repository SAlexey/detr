from argparse import ArgumentParser, Namespace
from models.detr import DETR, SetCriterion
from models.matcher import build_matcher

import torch
from models.transformer import build_transformer
from models.backbone import build_backbone
from typing import Any, List, Optional
import pytorch_lightning as pl
from torch import add
from torch.nn.modules.module import T_co 


class ModelBase(pl.LightningModule):

    def __init__(self, model, criterion, hparams) -> None:
        super.__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters(hparams)

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return self.model(*input)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.hparams.lr)

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

    @staticmethod
    def add_argparse_args(parents:List[ArgumentParser]=[]):
        parser = ArgumentParser(parents=parents, add_help=False)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--num_workers", type=int, default=1)
        return parser
        

class MeDeCl(ModelBase):
    def __init__(self, args) -> None:

        backbone = build_backbone(args)

        transformer = build_transformer(args)

        matcher = build_matcher(args)

        criterion = SetCriterion(
            args.num_classes, matcher, {
                "loss_ce": 1,
                "loss_bbox": args.bbox_loss_coef,
                "loss_giou": args.giou_loss_coef,
            }, eos_coef=args.eos_coef,
            losses = ['labels', 'boxes']
        )

        input_proj = (torch.nn.Conv2d if args.input_dim == "2d" else torch.nn.Conv3d)(
            backbone.num_channels, transformer.d_model, kernel_size=1
        )

        model = DETR(
            backbone, 
            input_proj,
            transformer, 
            args.num_classes, 
            args.num_queries, 
            aux_loss=args.aux_loss
        )

        super().__init__(model, criterion, args)

    def configure_optimizers(self):
        return torch.optim.AdamW([
            {"params": self.model.backbone.parameters(), "lr": self.hparams.lr_backbone},
            {"params": self.model.transformer.parameters(), "lr": self.hparams.lr_transformer}
        ], self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @staticmethod
    def add_argparse_args(parents: List[ArgumentParser]=[]):
        parser = super().add_argparse_args(parents)
        # * Model Parameters
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--num_queries", type=int, default=2)
        parser.add_argument("--input_dim", default="2d", type=str, options=["2d", "3d"])
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')
        # * Loss
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
        # * Matcher
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        # * Loss coefficients
        parser.add_argument('--bbox_loss_coef', default=5, type=float)
        parser.add_argument('--giou_loss_coef', default=2, type=float)
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        parser.add_argument('--lr_backbone', default=1e-3, type=float)
        parser.add_argument('--lr_transformer', default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        return parser





    


        



    
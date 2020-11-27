from collections import defaultdict
from argparse import Namespace
import json
from pathlib import Path
from typing import Callable, Dict, Optional, OrderedDict, Tuple
from pycocotools.coco import COCO

from typing_extensions import TypeAlias
from util.box_ops import box_xyxy_to_cxcywh, box_xyxy_to_cxcywh3d
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from util.misc import collate_fn


class NPZDatasetBase(Dataset):

    def __init__(self, items) -> None:
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.load(self.items[idx])


class DataModuleBase(pl.LightningDataModule):

    def __init__(self, hparams, dataset) -> None:
        self.hparams = hparams
        self.dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            shuffle=True, collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, collate_fn=collate_fn 
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, collate_fn=collate_fn 
        )

    def setup(self, stage: Optional[str]=None):
        root = Path(self.hparams.datadir)

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(list((root/"train").iterdir()))
            self.val_dataset = self.dataset(list((root/"val").iterdir()))

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(list((root/"test").iterdir()))
    
TransformType: TypeAlias = Callable[[Tuple[torch.Tensor, Dict]], Tuple[torch.Tensor, Dict]]

def default_annotation():
    return {
        "info": {},
        "images": [],
        "annotations": []
        }

class MRIDataset(NPZDatasetBase):

    def __init__(self, *args, transform:Optional[TransformType]=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform
        self._coco = {
            "sag": COCO(),
            "cor": COCO(),
            "axi": COCO()
        }
        
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
    
        image = torch.FloatTensor(item.get("image"))
        image_id = int(self.items[idx].stem)
        image_shape = image.shape[-3:]

        boxes = item.get("boxes") / (image_shape + image_shape)
        boxes = box_xyxy_to_cxcywh3d(torch.FloatTensor(boxes))

        target = {
            "image_id": image_id,
            "orig_size": image_shape,
            "labels": torch.FloatTensor(item.get("labels")),
            "boxes": boxes
        }

        if self.transform is not None:
            image, target = self.transform((image, target))
        return image, target

    @property
    def coco(self):
        return self._coco


class MRIDataModule(DataModuleBase):
    def __init__(self, hparams) -> None: 
        dataset = MRIDataset
        super().__init__(hparams, dataset)

    def setup(self, stage: Optional[str]):
        super().setup(stage)
        root = Path(self.hparams.datadir)

        for each in ["train", "val", "test"]:
            annotations = json.load(root / each /"annotation_by_plane.json")
            loader = getattr(self, f"{each}_dataloader")()
            for key, cocoset in loader.dataset.coco.items():
                cocoset.dataset = annotations[key]
                cocoset.createIndex()

def main():
    args = Namespace(
        datadir="/scratch/visual/ashestak/oai/v00/numpy/full",
        num_workers=1,
        batch_size=1
    )
    datamodule = MRIDataModule(args)
    datamodule.setup()
    



if __name__ == "__main__":
    main()

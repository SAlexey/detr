from __future__ import annotations
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
    
TransformType: TypeAlias = Callable[[Tuple[torch.Tensor, Dict]], Tuple[torch.Tensor, Dict]]


class MRIDataset(NPZDatasetBase):

    def __init__(self, *args, transform:Optional[TransformType]=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform
        self._coco = defaultdict(lambda:COCO())
        
    
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

    def setup(self, stage: Optional[str]):
        super().setup(stage)
        root = Path(self.hparams.datadir)

        is_valid = lambda p: p.suffix == ".npz"

        print("Getting items")
        test_items = [item for item in (root/"test").iterdir() if is_valid(item)]
        train_items = [item for item in (root/"train").iterdir() if is_valid(item)]
        val_items = train_items[-300:]
        train_items = train_items[:-300]

        self.test_dataset = MRIDataset(test_items)
        self.train_dataset = MRIDataset(train_items)
        self.val_dataset = MRIDataset(val_items)

        print("Creating Annotations")

        train_annotations = json.load(root/"train"/"annotations.json")
        config = train_annotations.pop("config", None)
        test_annotations = json.load(root/"test"/"annotations.json")
        config = test_annotations.pop("config", None) or config


        for key, val in test_annotations.items():
            print(f"Adding {key} annotations to test_annotations")
            coco = self.test_dataset.coco[key]
            coco.dataset = val 
            coco.createIndex()


        val_image_ids = [int(each.stem) for each in val_items]

        val_annotations = defaultdict(lambda: {
            "categories": train_annotations["sag"]["categories"],
            "images": [],
            "annotations": []
        })

        for key, train in train_annotations.items():
            print(f"Adding {key} annotations to train_annotations")
            for image in train["images"]:
                if image["id"] in val_image_ids:
                    val_annotations[key]["images"].append(image)
                    train["images"].remove(image)

            for annotation in train["annotations"]:
                if annotation["image_id"] in val_image_ids:
                    val_annotations[key]["annotations"].append(annotation)
                    train["anntotations"].remove(annotation)
            
            
            coco = self.train_dataset.coco[key]
            coco.dataset = train_annotations
            coco.createIndex()

        for key, val in val_annotations.items():
            print(f"Adding {key} annotations to val_annotations")
            coco = self.val_dataset.coco[key]
            coco.dataset = val
            coco.createIndex()

        print("COCO setup complete!")

        assert len(self.train_dataset), "train_dataset cannot be empty!"
        assert len(self.val_dataset), "val_dataset cannot be empty!"
        
        
        print(f"Training Items: {len(self.train_dataset)}")
        print(f"Validation Items: {len(self.val_dataset)}")
        print(f"Testing Items: {len(self.test_dataset)}")

        print("Data setup complete!")
        
        
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

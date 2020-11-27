from collections import defaultdict
import json
from pathlib import Path
from typing import Callable, Dict, Optional, OrderedDict, Tuple

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
    
TransformType: TypeAlias = Callable[[Tuple[torch.Tensor, Dict], Tuple[torch.Tensor, Dict]]]

class MRIDataset(NPZDatasetBase):

    def __init__(self, *args, transform:Optional[TransformType]=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform
        self._cocoAnn = defaultdict(dict([
            ("info", {}),
            ("images", [])
            ("annotations", []),
        ]))
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
    
        image = torch.FloatTensor(item.get("image")).unsqueeze(0)
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
    def coco_dict(self):
        planes = (
            ("sag", [1, 2, 4, 5]),
            ("cor", [0, 1, 3, 4]), 
            ("axi", [0, 2, 3, 5])
        )
        if self._cocoAnn is None:
            for i in range(len(self)):
                path = self.items[i]
                image, targets = self[i]
                image_id = int(path.stem)
                for plane, dims in planes:
                    annotation_set = self.cocoAnn[plane]
                    annotation_set["images"].append( {
                        "id" : image_id,
                        "width": image.size(dims[0]), 
                        "height": image.size(dims[1]), 
                        "file_name" : path 
                    })
                    for box, label in zip(targets["boxes"], targets["labels"]):
                        annotation_set["annotation"].append({
                            "id" : len(annotation_set["annotation"]), 
                            "image_id" : image_id, 
                            "category_id" : label, 
                            "area" : float, 
                            "bbox" : box_xyxy_to_cxcywh3d(box[dims]),  #[x,y,width,height]
                            "iscrowd" : 0,
                        })
        return self._cocoAnn


class MRIDataModule(DataModuleBase):
    def __init__(self, hparams) -> None: 
        dataset = MRIDataset
        super().__init__(hparams, dataset)

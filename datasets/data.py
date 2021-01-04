from __future__ import annotations

import json
import os
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import SimpleITK as sitk
import torch
import torchio as tio
from matplotlib.pyplot import box
from numpy.core.fromnumeric import ndim
from pycocotools.coco import COCO
from scipy import ndimage
from torch.nn.modules import sparse
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection, ImageFolder
from torchvision.datasets.folder import DatasetFolder
from torchvision import transforms as T
from tqdm import tqdm
from typing_extensions import TypeAlias
from util.box_ops import (convert, box_cxcywh_to_xyxy,
                          box_xyxy_to_cxcywh, box_xyxy_to_cxcywh3d,
                          box_zxyzxy_to_cdcxcydwh, box_zzxxyy_to_zxyzxy)
from util.misc import collate_fn
from .transforms import Compose, NormalizeBbox, SafeCrop


def is_np_file(path):
    return path.endswith(".npy") or path.endswith("npz")


class DetectionDataset(CocoDetection):

    """
    Copy paste with from torchvision.datasets 
     - changed image format 
     - added loader option
    """

    def __init__(self, loader:Callable[[str], Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = loader 

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """

        copy-paste from torchvisionn.datasets

        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.loader(os.path.join(self.root, path))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class NPYImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        loader = kwargs.pop("loader", np.load)
        is_valid_file = kwargs.pop("is_valid_file", is_np_file)
        super().__init__(*args, loader=loader, is_valid_file=is_valid_file, **kwargs)


class NPZDatasetBase(Dataset):

    def __init__(self, items) -> None:
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        with np.load(self.items[idx]) as item:
            return dict(item)


class DataModuleBase(pl.LightningDataModule):

    def __init__(self, hparams, *args,  collate_fn=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.collate = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            shuffle=True, collate_fn=self.collate, num_workers=self.hparams.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, collate_fn=self.collate, num_workers=self.hparams.num_workers 
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, collate_fn=self.collate, num_workers=self.hparams.num_workers
        )

    
TransformType: TypeAlias = Callable[[Tuple[torch.Tensor, Dict]], Tuple[torch.Tensor, Dict]]


class MRIDataset(NPZDatasetBase):

    def __init__(self, *args, transform:Optional[TransformType]=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform
        self._coco = defaultdict(lambda:COCO())

        
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
    
        image = torch.tensor(item.get("image"), dtype=torch.float32)
        image_id = int(self.items[idx].stem)
        image_shape = image.shape[-3:]

        boxes = torch.tensor(item.get("boxes"), dtype=torch.float32)
        boxes = self.formatter.convert(boxes, "xxyy", "ccwh")
        boxes = boxes / torch.tensor((image_shape + image_shape), dtype=torch.float32)

        target = {
            "image_id": image_id,
            "orig_size": image_shape,
            "labels": torch.LongTensor(item.get("labels")),
            "boxes": boxes
        }

        if self.transform is not None:
            image, target = self.transform((image, target))

        return image, target

    @property
    def coco(self):
        return self._coco


class _MRISliceDataset(Dataset):

    def __init__(self, items, transform=None):
        super().__init__()
        self.items = items
        self.transform = transform

    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image, target = self.items[idx]
        if self.transform is not None:
            image, target = self.transform(image, target)
        target["labels"] = target["labels"][0].unsqueeze(0)
        target["boxes"] = target["boxes"].unsqueeze(0).float()
        image = torch.as_tensor(image).unsqueeze(0)
        print(target)
        return image, target



    

class OverfitAugData(DataModuleBase):

    def __init__(self, *args, train_transforms=None, val_transforms=None, test_transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_transforms = Compose([SafeCrop(300, 300), NormalizeBbox(300, 300)])
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms


    def setup(self, stage=None):
        items = []

        for item in Path(self.hparams.datadir).iterdir():
            with np.load(item) as f:
                items.append((torch.as_tensor(f["image"], dtype=torch.float32), {"iscrowd": 0, "boxes": f["bbox"], "labels": torch.as_tensor([0])}))

        
        self.train_dataset = _MRISliceDataset(items, transform=getattr(self, "train_transforms", None))
        self.val_dataset = _MRISliceDataset(items, transform=getattr(self, "train_transforms", None))


class _MRISliceData(DataModuleBase):

    def setup(self, stage=None):
        sourcePath = Path("/vis/scratchN/oaiDataBase/v00/OAI/")
        segmntPath = Path("/vis/scratchN/bzftacka/OAI_DESS_Data_AllTPs/Merged/v00/OAI")
        stat = pd.read_csv(sourcePath/"statistics/SAG_3D_DESS_LEFT", sep=" ", header=None)
        stat["id"] = stat[0].str.extract(r"(\d{7})")
        stat["side"] = stat[1].map(lambda s: s.split("_")[-1]).str.lower()
        stat.rename({0: "path"}, axis=1, inplace=True)
        stat.drop([1, 2], axis=1, inplace=True)
        stat.path = stat.path.map(Path)
        imgp = lambda row: sourcePath/row.path
        sgmp = lambda row: segmntPath/row.path/"Segmentation" 

        reader = sitk.ImageSeriesReader()
        (_, row) = next(stat.iterrows())

        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(imgp(row))))
        image = reader.Execute()
        image = sitk.GetArrayFromImage(image)
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(sgmp(row))))
        segmentation = reader.Execute()
        segmentation = sitk.GetArrayFromImage(segmentation)

        print(image.shape)
        print(segmentation.shape)

        (*_, (z1, *_), (z2, *_)) = ndimage.find_objects(segmentation)

        indices = list(np.random.randint(z1.start + 10, z1.stop - 10, 2)) + list(np.random.randint(z2.start + 10, z2.stop - 10, 2))

        items = []

        for each in indices:
            image_slice = image[each]
            mask_slice = segmentation[each]

            (*_, (ys, xs)) = ndimage.find_objects(mask_slice)


            bbox = np.array([xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start])

            items.append((
                torch.as_tensor(image_slice.astype(np.int16)).unsqueeze(0),
                {"boxes": torch.as_tensor(bbox), "labels": torch.as_tensor([0]),"iscrowd": torch.as_tensor([0])}
            ))

        self.train_dataset = _MRISliceDataset(items)
        self.val_dataset = _MRISliceDataset(items)

class MRISliceDetectionData(DataModuleBase):

    def __init__(self, *args, **kwargs):
        collate = kwargs.pop("collate_fn", collate_fn)
        super().__init__(*args, collate_fn=collate, **kwargs)

    def setup(self, stage=None):
        root = Path(self.hparams.datadir)
        is_valid = lambda path: path.suffix == ".npz"
        for each in ("train", "val"):
            path = root/each

            for item in path.iterdir():
                if is_valid(item):
                    pass
                    






class MRISliceDataset(MRIDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_ids = set()
        self.images = []
        self.annotations = []
        self.categories = []

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)
        target["orig_size"] = target["orig_size"][1:]
        
        # take the first meniscus label
        target["labels"] = target["labels"][0]
        
        # take the fist meniscus bounding box
        tgt_box = target["boxes"][0]
        
        # just take the x-y plane of the box
        target["boxes"] = tgt_box[[1, 2, 4, 5]] # box = [x1, y1, width, height] relative
 
        # mid-section of the box
        box_center = int(np.round(tgt_box[0] * 160))

        # add some variance
        slice_index = np.random.randint(box_center - 3, box_center + 3)


        # build coco dataset for evaluation
        if target["image_id"] not in self.image_ids:
            # add image to annotations if it is not there already
            self.image_ids.add(target["image_id"])
            self.images.append({
                "id": target["image_id"],
                "width": 300,
                "height": 300,
            })
           
            # rescale box back to image size
            # convert to [x1, y1, x2, y2]
            box = convert(target["boxes"], "ccwh", "xyxy") * np.array((300,)*4)

            self.annotations.append({
                "id": len(self.annotations),
                "image_id": target["image_id"],
                "category_id": target["labels"].item(),
                "iscrowd": 0,
                "area": (box[2] - box[0]) * (box[3] - box[1]),
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]] # box = [x, y, width, height]
            })

            if target["labels"].item() not in [cat.get("id") for cat in self.categories]:
                self.categories.append({"id": target["labels"].item(), "name": "meniscus"})
        target["boxes"] = target["boxes"].unsqueeze(0)
        target["labels"] = target["labels"].unsqueeze(0)
        return image[slice_index].unsqueeze(0), target



class LitBackboneDataSet(NPZDatasetBase):

    def __init__(self, items):
        self.items = items
        self.n_features = lambda a: ndimage.label(a)[1]

    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):

        with np.load(self.items[idx]) as item:
            image = item["image"]
            segmentation = item["segmentation"]
        
        menisci = np.isin(segmentation, (5,6))

        num_features = [self.n_features(s) for s in menisci]
        num_features = torch.tensor(num_features, dtype=torch.long).clip(0, 2)

        indices = np.arange(len(num_features))[10:150]

        np.random.shuffle(indices)

        indices = indices[:80]

        if np.random.rand() >= 0.5:
            noize = np.random.rand(*image.shape)
            menisci[::2, :, :] = True
            noize[menisci] = 1

            image = image * noize

        image = torch.tensor(image, dtype=torch.float32)

        return image[indices], num_features[indices]


class LitBackboneData(DataModuleBase):

    def setup(self, stage=None):
        root = Path(self.hparams.datadir)
        path = root.iterdir()

        items = []

        pbar = tqdm(total=550)
        while len(items) < 550:
            p = next(path)
            try: 
                with np.load(p) as item:
                    items.append(p)
                    pbar.update()
            except Exception as e:
                print(e)
                continue

        train_items = items[:500]
        val_items = items[500:]

        self.train_dataset = LitBackboneDataSet(train_items)
        self.val_dataset = LitBackboneDataSet(val_items)    

class SimpleOverfitWithAugDataModule(DataModuleBase):

    def setup(self, stage=None):

        root = Path("/scratch/visual/ashestak/detr/datasets/aug_sample")

        train_dataset = MRISliceDataset(train_items, tra)

class MRISliceDataModule(DataModuleBase):

    def setup(self, stage=None):
        root = Path(self.hparams.datadir)

        is_valid = lambda p: p.suffix == ".npz"

        train_items = [item for item in (root/"train").iterdir() if is_valid(item)]
        val_items = [item for item in (root/"test").iterdir() if is_valid(item)]
        
        self.train_dataset = MRISliceDataset(train_items)
        self.val_dataset = MRISliceDataset(val_items)


class MRIDataModule(DataModuleBase):

    def setup(self, stage: Optional[str]=None):
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
        datadir="/scratch/visual/ashestak/oai/v00/numpy/full/train",
        num_workers=1,
        batch_size=1
    )
    datamodule = LitBackboneData(args)
    datamodule.setup()

    for item in datamodule.train_dataloader():
        item
    



if __name__ == "__main__":
    main()

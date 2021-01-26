
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torchio as tio
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchio.data.image import LabelMap
from torchvision import ops
from util.box_ops import mask_to_bbox
from util.misc import nested_tensor_from_tensor_list
from torchvision.datasets import CocoDetection
import albumentations as A

import datasets.transforms as TT

T = TypeVar("T")
TrFunc = Callable[[T], T]
SetupFunc = Callable[[pl.LightningDataModule, Optional[str]], None]

DICOM_SRC = Path("/scratch/visual/ashestak/oai/v00/dicom/")
NUMPY_SRC = Path("/scratch/visual/ashestak/oai/v00/numpy/")


class TransformableSubset(Dataset):
    """
    A Wrapper for PyTorch Subset that can take optinal transforms
    
    Arguments: 
        subset (pytroch Subset) Subset of a dataset at specified indices.
            see https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Subset
        transforms (callable) transforms to be applied to the items of the subset 
    """

    def __init__(self, subset: Subset, transform:Optional[TrFunc] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        item = self.subset[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item 

    def __len__(self):
        return len(self.subset)


class SubjectsDataset(Dataset):

    def __init__(self, items, transform=None):
        self.items = items 
        self.transform = transform


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        item = np.load(self.items[idx])

        image = tio.ScalarImage(tensor=item["image"], affine=item["affine"], spacing=item["spacing"])
        image._loaded = True

        mask = tio.LabelMap(tensor=item["mask"], affine=item["affine"], spacing=item["spacing"])
        mask._loaded = True

        subject = tio.Subject(image=image, mask=mask, id=str(item["id"]), side=str(item["side"]).lower())

        if self.transform is not None:
            subject = self.transform(subject)

        return subject


class MenisciDataset(Dataset):

    def __init__(self, paths,  transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = np.load(self.paths[idx])

        mask = item["mask"].isin((5,6))

        image = torch.as_tensor(item["image"], dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.long)

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        if mask.dtype != torch.long:
            mask = mask.long()

        mask = mask.isin((5,6))

        objetcs = ndimage.find_objects(mask, max_label=6)
        
        




class Subjects(TransformableSubset):

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        w, h = item.image.shape[-2:]
        
        image = item.image.data.squeeze()
        mask = item.mask.data.squeeze().long()

        obj, *_ = next(o for o in ndimage.find_objects(mask) if o is not None)
        slice = (obj.start + obj.stop) // 2
        
        image = image[slice, ...]
        mask = mask[slice, ...]

        tgt, bbox = mask_to_bbox(mask, bbox_fmt="ccwh")
        image = image.unsqueeze(0)
        bbox = bbox / torch.as_tensor((w, h, w, h), dtype=torch.float32)

        return image, { "boxes": bbox, "labels": tgt, "num_objects": len(tgt) }


class SegmentedKMRIOAIv00(pl.LightningDataModule):         

    """
    Osteoarthritis Initiative MRI Data Segmented by Ambellan et al.
        
    """     

    def __init__(
            self, 
            *args, 
            transforms: Optional[List[Callable[[Any], Any]]] = [],
            collate_fn: Optional[List[Callable[[Any], Any]]] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            **kwargs
        ):

        """
        Initializes the data module
        Parameters:
            batch_size:     (int) batch size used in the DataLoader
            num_workers:    (int) number of processes used in the DataLoader
            collate_fn:     (list[callable]) function that collates the batch in the DataLoader
            transforms:     (list[callable]) transforms on ALL splits

            train_transforms: (callable) transforms ONLY on train split
            val_transforms:   (callable) transforms ONLY on val split
            test_transforms:  (callable) transforms ONLY on test split
        """

        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

        self.train_transforms = self.train_transforms or []
        self.val_transforms = self.val_transforms or []
        self.test_transforms = self.test_transforms or []

        assert isinstance(self.train_transforms, list)
        assert isinstance(self.val_transforms, list)
        assert isinstance(self.test_transforms, list)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True, 
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
            batch_size=self.batch_size,
            colalte_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False
        )

    def setup(self, stage=None):

        index = pd.read_csv("/scratch/visual/ashestak/oai/v00/numpy/index.csv", header=[0])["0"]
        index = index.str.cat(["/subject.npz",] * len(index))
        

        transforms = tio.Compose([
            LRFlip(),                                        # flips left knee to right 
            tio.RemapLabels({1:0, 2:0, 3:0, 4:0, 5:1, 6:2}), # only one meniscus is nonzero
        ] + self.train_transforms)

        train_transforms = tio.Compose([
            tio.RandomAffine(
                p=0.5,  
                degrees=(90, 90, 0, 0, 0, 0)                # rotate 90 deg around sag-axis
            ),
            tio.RandomAffine(
                p=0.5,
                degrees=(5, 0, 0),                          # rotate +/- 5 deg around sag-axis
                translation=(0, 2, 2),                      # translate +/- 2 mm in the non sag-planes
                scales=(0.9, 1.2),                          # rescale between 0.9 - 1.2 
                isotropic=True,
                image_interpolation="nearest"  
            ),
            tio.RandomFlip(axes=("AP", "IS")),              # Anterior-Posterior Interior-Superior flips
            tio.Crop((0, 42, 42)),                          # crop image to (160, 300, 300) 
        ] + self.train_transforms)

        val_transforms = tio.Compose([
            tio.Crop((0, 42, 42)),                          # crop image to (160, 300, 300)
        ] + self.val_transforms)

        full_dataset = SubjectsDataset(index, transform=transforms)

        num_total = len(full_dataset)
        num_test = int(round(num_total * 0.25))
        num_val = int(round(num_total * 0.05))
        num_train = num_total - num_test - num_val

        print(f"Total: {num_total}", f"Train: {num_train}", f"Val: {num_val}", f"Test: {num_test}", sep="\n")

        train_subset, val_subset, test_subset = random_split(full_dataset, (num_train, num_val, num_test))

        self.train_dataset = Subjects(train_subset, transform=train_transforms)
        self.val_dataset = Subjects(val_subset, transform=val_transforms)
        self.test_dataset = Subjects(test_subset, transform=val_transforms)


def subject_from_row(row:Dict[str, Any]):
    
    return tio.Subject(
        image=tio.ScalarImage(str(DICOM_SRC / row["path"] / "Image")),
        mask=tio.LabelMap(str(DICOM_SRC / row["path"] / "Segmentation")), 
        side=row["side"].lower(),
        id=row["id"]
    )


def subjects_from_dicom(num_workers=1, ignore_paths=[]):
    """
        Reads the following files:

        /scratch/visual/ashestak/oai/v00/dicom/statistics/SAG_3D_DESS_LEFT
        /scratch/visual/ashestak/oai/v00/dicom/statistics/SAG_3D_DESS_RIGHT

        concatennates them together into a single pandas dataframe
        creates a list of subjects containing keys
         - image:  mri images
         - mask: segmentation
         - id: patient id
         - side: leg side
    """  

    statLeft = pd.read_csv(DICOM_SRC/"statistics/SAG_3D_DESS_LEFT", sep=" ", header=None)
    statRight = pd.read_csv(DICOM_SRC/"statistics/SAG_3D_DESS_RIGHT", sep=" ", header=None)

    stat = statLeft.append(statRight)

    stat["id"] = stat[0].str.extract(r"(\d{7})")
    stat["side"] = stat[1].str.extract(r"([A-Z]+$)")
    
    stat.rename({0: "path"}, axis=1, inplace=True)
    stat.mask(stat["path"].isin(ignore_paths), inplace=True)
    stat.dropna(inplace=True, subset=["path"])
    

    print(f"Preparing Subjects (num_proc={num_workers})")
    with mp.Pool(num_workers) as pool:
        subjects = list(pool.imap_unordered(subject_from_row, stat.to_dict(orient="records"), chunksize=500))
    print("Done!")
    return subjects


def _slice_loader_fn(coco, img_id):
    obj = coco.loadImgs(img_id)[0]
    img = np.load(obj["file_name"])[obj["image_key"]]
    img = np.squeeze(img)[obj["slice"]]
    img = np.stack((img,)*3)
    img = np.transpose(img, (1, 2, 0))
    return img    


def _volume_loader_fn(coco, img_id):
    obj = coco.loadImgs(img_id)[0]
    img = np.load(obj["file_name"])[obj["image_key"]]
    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))
    return img

class LitDetectionSet(CocoDetection):
    
    def __init__(self, *args, image_loader_fn=_slice_loader_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_image = image_loader_fn
    
    def __getitem__(self, index:int):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img = self.load_image(coco, img_id)
        tgt = coco.loadAnns(ann_ids)
        
        tgt = {'image_id': img_id, 'annotations': tgt}
        img, tgt = self.prepare(img, tgt)


        if self.transforms is not None:
            img, tgt = self.transforms(img, tgt)

        return img, tgt


    def prepare(self, img, tgt):

        h, w, _ = img.shape
        anno = tgt.pop("annotations")

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        bboxes = [obj["bbox"] for obj in anno]
        labels = [obj["category_id"] for obj in anno]
        areas = [obj["area"] for obj in anno]
        iscrowd = [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        bboxes = ops.box_convert(bboxes, "xywh", "xyxy")

        keep = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
        
        tgt["boxes"] = bboxes[keep]
        tgt["labels"] = labels[keep]
        tgt["area"] = areas[keep]
        tgt["iscrowd"] = iscrowd[keep]

        tgt["orig_size"] = torch.as_tensor([int(h), int(w)])
        tgt["size"] = torch.as_tensor([int(h), int(w)])
        
        return img, tgt


class LitDetectionData(pl.LightningDataModule):

    def __init__(
            self, 
            *args, 
            collate_fn: Optional[List[Callable[[Any], Any]]] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            **kwargs
        ):

        """
        Initializes the data module
        Parameters:
            batch_size:     (int) batch size used in the DataLoader
            num_workers:    (int) number of processes used in the DataLoader
            collate_fn:     (list[callable]) function that collates the batch in the DataLoader
            transforms:     (list[callable]) transforms on ALL splits

            train_transforms: (callable) transforms ONLY on train split
            val_transforms:   (callable) transforms ONLY on val split
            test_transforms:  (callable) transforms ONLY on test split
        """

        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True, 
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
            batch_size=self.batch_size,
            colalte_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False
        )       

    def setup(self, stage=None):

        transforms = TT.Compose([
            TT.ToTensor(), 
            TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        root = "/scratch/visual/ashestak/oai/v00/numpy"
        anns = "/scratch/visual/ashestak/detr/m_slice_anns.json"

        dataset = LitDetectionSet(root, anns,  transforms=transforms)
       
        num_total = len(dataset)
        num_test = int(round(num_total * 0.2))
        num_val = int(round(num_total * 0.1))
        num_train = num_total - num_test - num_val

        train, val, test = random_split(dataset, [num_train, num_val, num_test])

        print(
            f"Total: {num_total}", 
            f"Train: {num_train}", 
            f"Val:   {num_val}", 
            f"Test:  {num_test}", 
            sep="\n============\n"
        )

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
    
        
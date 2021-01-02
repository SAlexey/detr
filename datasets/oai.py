from typing import Any, Callable, Generic, Optional, TypeVar
import warnings
import torch
import torchio as tio 
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Subset, random_split as torch_random_split
from tqdm.std import tqdm
from enum import Enum

T = TypeVar("T")
TrFunc = Callable[[T], T]
SetupFunc = Callable[[pl.LightningDataModule, Optional[str]], None]


def collate_fn(batch):
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

class TransformableSubset(Dataset):
    """
    A Wrapper for PyTorch Subset that can take optinal transforms
    
    Arguments: 
        subset (pytroch Subset) Subset of a dataset at specified indices.
            see https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Subset
        transforms (callable) transforms to be applied to the items of the subset 
    """

    def __init__(self, subset: Subset, transform:Optional[TrFunc] = None) -> None:
        if not isinstance(subset, Subset):
            raise ValueError(f"`subset` must be an instance of Subset, got {type(subset).__name__}")


        if hasattr(subset.dataset, "transforms") and transform is not None:
            warnings.warn(
                f"""You have provided transforms to be applied to the ({type(subset.dataset).__name__}) 
                but it already has transforms defined! 
                Make sure this is intended, otherwise remove either of the transforms""")
        
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        item = self.subset[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item 

    def __len__(self):
        return len(self.subset)


class OAIFullKMRIDetectionDataset(tio.SubjectsDataset):
    """
    Anatomical Structure Detection Dataset
    from Full Knee MRI
    """

    def __getitem__(self, idx):
        subject = super().__getitem__(idx)
        label_map = subject["label_map"][tio.DATA]
        objects = ndimage.find_objects(label_map)

        bboxes = []
        labels = []

        for label, (*_, zs, ys, xs) in enumerate(objects):
            bboxes.append([zs.start, xs.start, ys.start, zs.stop, xs.stop, ys.stop])
            labels.append(label)

        subject["labels"] = labels
        subject["boxes"] = bboxes
             
        return subject

class OAISliceKMRIDetectionDataset(OAIFullKMRIDetectionDataset):

    def __getitem__(self, idx):
        subject = super().__getitem__(idx)
        obj_id = 5 if subject["side"] == "left" else 6
        bbox = subject["boxes"][obj_id]
        label = obj_id


class OAIFullKMRIDetectionSubset(TransformableSubset):

    def __init__(self, *args, extra_transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_transforms = extra_transforms

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.extra_transforms is not None:
            item = self.extra_transforms(item)
        return item["image"][tio.DATA], {"boxes": item["boxes"], "labels": item["labels"]}



class OAIMRI(pl.LightningDataModule):         

    """Osteoarthritis Initiative MRI Data Module"""     

    def __init__(
            self, 
            *args, 
            collate_fn: Optional[Callable[[Any], Any]] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            shuffle: bool = True,
            **kwargs
        ):

        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=self.shuffle, 
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

        subjects = subjects_from_dicom()

        full_dataset = tio.SubjectsDataset(subjects)

        total_len = len(subjects)
        train_len = int(round(total_len * 0.65))
        test_len = total_len - train_len 
        val_len = int(round(train_len * 0.05))
        train_len -= val_len
        
        assert sum((train_len, val_len, test_len)) == total_len

        
        print(f"# Train Subjects: {train_len}")
        print(f"# Val Subjects: {val_len}")
        print(f"# Test Subjects: {test_len}")

        train_subset, val_subset, test_subset = random_split(full_dataset, [train_len, val_len, test_len])

        train_subset.transform = self.train_transforms
        val_subset.transform = self.val_transforms
        test_subset.transform = self.test_transforms

        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_subset

def random_split(*args, subset_class=TransformableSubset, **kwargs):
    return [subset_class(each) for each in torch_random_split(*args, **kwargs)]

def subjects_from_dicom():
    src_img = Path("/vis/scratchN/oaiDataBase/v00/OAI/")
    src_msk = Path("/vis/scratchN/bzftacka/OAI_DESS_Data_AllTPs/Merged/v00/OAI")
    # get image path from the row
    
    imgp = lambda row: src_img/row.path

    # get segmentation path from the row
    sgmp = lambda row: src_msk/row.path/"Segmentation"

    statLeft = pd.read_csv(src_img/"statistics/SAG_3D_DESS_LEFT", sep=" ", header=None)
    statRight = pd.read_csv(src_img/"statistics/SAG_3D_DESS_RIGHT", sep=" ", header=None)

    # dataframe with all paths
    stat = statLeft.append(statRight)

    #extract patient id from path
    stat["id"] = stat[0].str.extract(r"(\d{7})")

    # extract side from the file name
    stat["side"] = stat[1].map(lambda s: s.split("_")[-1]).str.lower()

    # clean up 
    stat.rename({0: "path"}, axis=1, inplace=True)
    stat.drop([1, 2], axis=1, inplace=True)
    stat.path = stat.path.map(Path)

    # prepare items
    subjects = []

    failed = pd.read_csv("/scratch/visual/ashestak/oai/v00/numpy/full/failed.csv")
    failed = failed.groupby(["Unnamed: 0"]).get_group("path")["0"].values

    for _, row in tqdm(stat.iterrows(), total=len(stat), desc="Preparing Subjects"):
        if row["path"] in failed:
            continue

        subject = tio.Subject(
            image=tio.ScalarImage(imgp(row)),
            label_map=tio.LabelMap(sgmp(row)),
            side=row["side"],
            id=row["id"]
        )

        subjects.append(subject)

    return subjects


def main():

    dm = OAIMRI(
        dataset_class=OAIFullKMRIDetectionDataset, 
        subset_class=OAIFullKMRIDetectionSubset, 
        collate_fn=collate_fn
    )

    dm.setup()

    item = next(iter(dm.train_dataloader()))
    item

if __name__ == "__main__":
    main()


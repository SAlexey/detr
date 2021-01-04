
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import torch
import torchio as tio 
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from scipy import ndimage
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from multiprocessing import Pool
from util.misc import nested_tensor_from_tensor_list
from .transforms import *

T = TypeVar("T")
TrFunc = Callable[[T], T]
SetupFunc = Callable[[pl.LightningDataModule, Optional[str]], None]



def collate_subjects(batch):
    images = torch.cat([s["image"][tio.DATA] for s in batch])
    images = nested_tensor_from_tensor_list(images)
    
    
    pass


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
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
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        item = self.subset[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item 

    def __len__(self):
        return len(self.subset)

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
        
        failed = pd.read_csv("/scratch/visual/ashestak/oai/v00/numpy/full/failed.csv")
        failed = failed.groupby(["Unnamed: 0"]).get_group("path")["0"].values

        subjects = subjects_from_dicom(num_workers=self.num_workers, ignore_paths=failed)

        transforms = tio.Compose([
            LRFlip(),                                        # flips left knee to right 
            tio.ToCanonical(),                               # puts axis defining sag-plane first
            tio.RescaleIntensity(percentiles=[0.5, 99.5]),   # rescales intensity to be in [0, 1]
            tio.ZNormalization(),                            # sets mean 0 and std 1 
            tio.RemapLabels({1:0, 2:0, 3:0, 4:0, 5:1, 6:0}), # only one meniscus is nonzero
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
                isotropic=True  
            ),
            tio.RandomFlip(axes=("AP", "IS")),              # Anterior-Posterior Interior-Superior flips
            tio.Crop((0, 42, 42)),                          # crop image to (160, 300, 300) 
            ObjectSlice(obj_id=1),                          # get middle slice of meniscus
            LabelMapToBbox(),                               # extracts bbox 
            NormalizeBbox(scale_fct=(300, 300))             # scale bbox by the image size
        ] + self.train_transforms)

        val_transforms = tio.Compose([
            tio.Crop((0, 42, 42)),                          # crop image to (160, 300, 300) 
            ObjectSlice(obj_id=1),                          # get middle slice of meniscus
            LabelMapToBbox(),                               # extracts bbox 
            NormalizeBbox(scale_fct=(300, 300))             # scale bbox by the image size
        ] + self.val_transforms)

        full_dataset = tio.SubjectsDataset(subjects, transform=transforms)

        num_total = len(full_dataset)
        num_test = int(round(num_total * 0.25))
        num_val = int(round(num_total * 0.05))
        num_train = num_total - num_test - num_val

        print(f"Total: {num_total}", f"Train: {num_train}", f"Val: {num_val}", f"Test: {num_test}", sep="\n")

        train_subset, val_subset, test_subset = random_split(full_dataset, (num_train, num_val, num_test))

        self.train_dataset = TransformableSubset(train_subset, transform=train_transforms)
        self.val_dataset = TransformableSubset(val_subset, transform=val_transforms)
        self.test_dataset = TransformableSubset(test_subset, transform=val_transforms)


def subjects_from_dicom(num_workers=1, ignore_paths=[]):
    """
        Reads the following files:

        /vis/scratchN/oaiDataBase/v00/OAI/statistics/SAG_3D_DESS_LEFT
        /vis/scratchN/oaiDataBase/v00/OAI/statistics/SAG_3D_DESS_RIGHT

        concatennates them together into a single pandas dataframe
        creates a list of subjects containing keys
         - image:  mri images
         - label_map: segmentation
         - id: patient id
         - side: leg side
    """

    src_img = Path("/vis/scratchN/oaiDataBase/v00/OAI/")
    src_msk = Path("/vis/scratchN/bzftacka/OAI_DESS_Data_AllTPs/Merged/v00/OAI")

    statLeft = pd.read_csv(src_img/"statistics/SAG_3D_DESS_LEFT", sep=" ", header=None)
    statRight = pd.read_csv(src_img/"statistics/SAG_3D_DESS_RIGHT", sep=" ", header=None)

    stat = statLeft.append(statRight)

    stat["id"] = stat[0].str.extract(r"(\d{7})")
    stat["side"] = stat[1].map(lambda s: s.split("_")[-1]).str.lower()
    
    stat.rename({0: "path"}, axis=1, inplace=True)
    stat["path"] = stat["path"].map(Path)

    stat["image"] = stat["path"].map(lambda p: tio.ScalarImage(src_img/p))
    stat["label_map"] = stat["path"].map(lambda p: tio.LabelMap(src_msk/p/"Segmentation"))

    stat.mask(stat["path"].isin(ignore_paths), inplace=True)
    stat.dropna(inplace=True, subset=["path"])
    stat.drop([1, 2, "path"], axis=1, inplace=True)

    print(f"Preparing Subjects (num_proc={num_workers})")
    with Pool(num_workers) as pool:
        subjects = list(pool.imap_unordered(tio.Subject, stat.to_dict(orient="records"), chunksize=500))
    print("Done!")
    return subjects
    



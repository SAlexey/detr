
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import torch
import torchio as tio 
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from scipy import ndimage
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from multiprocessing import Pool


T = TypeVar("T")
TrFunc = Callable[[T], T]
SetupFunc = Callable[[pl.LightningDataModule, Optional[str]], None]

class LabelMapToBbox(tio.transforms.Transform):

    """
        Extracts object bounding boxes from a label map
        optionally filtering them depending on the obj. id    
        and optionally re-labels them to (starting from 0) 

        the ordering of the objects in the label map is preserved

        boxes are in format xyxy i.e [z0, y0, x0, z1, y1, x1] 
    """

    def __init__(self, *args, keep_labels=[4, 5], reset_labels=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.args_names = []
        self.keep_labels = keep_labels
        self.reset_labels = reset_labels


    def apply_transform(self, subject: tio.Subject):
        label_map = subject["label_map"][tio.DATA].squeeze() # remove channel dimension
        objects = ndimage.find_objects(label_map, max_label=max(self.keep_labels)) 
        objects = (objects[keep] for keep in self.keep_labels)

        objects = enumerate(objects) if self.reset_labels else zip(self.keep_labels, objects)
        
        bboxes = []
        labels = []

        for label, (zs, ys, xs) in objects:
            bboxes.append([zs.start, xs.start, ys.start, zs.stop, xs.stop, ys.stop])
            labels.append(label)

        subject["labels"] = torch.as_tensor(labels, dtype=torch.long)
        subject["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        return subject


class LRFlip(tio.transforms.Transform):

    def __init__(self, *args, side="left", **kwargs):
        super().__init__(*args, **kwargs)
        self.args_names = []
        self.flip = tio.transforms.RandomFlip(axes=("LR",), flip_probability=1)

    def apply_transform(self, subject: tio.Subject):
        if subject["side"][0] == "left":
            subject = self.flip(subject)
        return subject

def collate_subjects(batch):
    images = torch.stack[[s["image"][tio.DATA] for s  in batch]]
    
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
            transforms: Optional[Callable[[Any], Any]] = None,
            collate_fn: Optional[Callable[[Any], Any]] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            **kwargs
        ):

        """
        Initializes the data module
        Parameters:
            batch_size:     (int) batch size used in the DataLoader
            num_workers:    (int) number of processes used in the DataLoader
            collate_fn:     (callable) function that collates the batch in the DataLoader
            transforms:     (callable) transforms on ALL splits

            train_transforms: (callable) transforms ONLY on train split
            val_transforms:   (callable) transforms ONLY on val split
            test_transforms:  (callable) transforms ONLY on test split
        """

        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

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
        """
        Reads the following files:

        /vis/scratchN/oaiDataBase/v00/OAI/statistics/SAG_3D_DESS_LEFT
        /vis/scratchN/oaiDataBase/v00/OAI/statistics/SAG_3D_DESS_RIGHT

        concatennates them together into a single pandas dataframe
        splits the paths into train test and val subsets 
        """
        failed = pd.read_csv("/scratch/visual/ashestak/oai/v00/numpy/full/failed.csv")
        failed = failed.groupby(["Unnamed: 0"]).get_group("path")["0"].values

        subjects = subjects_from_dicom(num_workers=self.num_workers, ignore_paths=failed)

        full_dataset = tio.SubjectsDataset(subjects, transform=self.transforms)

        num_total = len(full_dataset)
        num_test = int(round(num_total * 0.3))
        num_val = int(round((num_total - num_test) * 0.1))
        num_train = num_total - num_test - num_val

        print(f"Total: {num_total}", f"Train: {num_train}", f"Val: {num_val}", f"Test: {num_test}", sep="\n")

        train_subset, val_subset, test_subset = random_split(full_dataset, (num_train, num_val, num_test))

        self.train_dataset = TransformableSubset(train_subset, transform=self.train_transforms)
        self.val_dataset = TransformableSubset(val_subset, transform=self.val_transforms)
        self.test_dataset = TransformableSubset(test_subset, transform=self.test_transforms)

        train_subset.transform = self.train_transforms
        val_subset.transform = self.val_transforms
        test_subset.transform = self.test_transforms

        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_subset


def subjects_from_dicom(num_workers=1, ignore_paths=[]):
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
    

def main():

    transforms = tio.Compose([
        LRFlip(),
        LabelMapToBbox(),
        tio.transforms.RescaleIntensity(percentiles=[0.5, 99.5]),
        tio.transforms.ZNormalization()
    ])

    dm = SegmentedKMRIOAIv00(num_workers=5, transforms=transforms, batch_size=4)

    dm.setup()

    loader = dm.train_dataloader()
    item = next(iter(loader))
    item

if __name__ == "__main__":
    main()


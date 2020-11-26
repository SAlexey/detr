from pathlib import Path
from typing import Optional
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
        
        train_items = list((root/"train").iterdir())
        val_items = list((root/"val").iterdir())
        test_items = list((root/"test").iterdir())

        self.train_dataset = self.dataset(train_items)
        self.val_dataset = self.dataset(val_items)
        self.test_dataset = self.dataset(test_items)
        

class MRIDataset(NPZDatasetBase):

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        inputs = torch.FloatTensor(item.get("image")).unsqueeze(0)
        labels = torch.FloatTensor(item.get("labels"))
        boxes = torch.FloatTensor(item.get("boxes"))

        return inputs, {"labels": labels, "boxes": boxes}


class MRIDataModule(DataModuleBase):
    def __init__(self, hparams) -> None: 
        dataset = MRIDataset
        super().__init__(hparams, dataset)

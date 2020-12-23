import pytorch_lightning as pl
import torchio as tio
import albumentations as A 
 


class SliceDataModule(pl.LightningDataModule):

    datadir: str

    def __init__(
        self, 
        datadir: str,
    ):
        super().__init__()
        

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
                        
            train_subjects = []
            train_dataset = tio.SubjectsDataset(train_subjects)

            val_subjects = []
            val_dataset = tio.SubjectsDataset(val_subjects)
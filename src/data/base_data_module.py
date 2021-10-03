from typing import Type, Any

from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision import transforms


class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 dataset: Type[Any],
                 train_transform: Type[Any],
                 test_transform: Type[Any],
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_root: str = 'data',
                 valid_ratio: float = 0.1):
        """
        Base Data Module

        :arg
            Dataset: Enter Dataset
            batch_size: Enter batch size
            num_workers: Enter number of workers
            size: Enter resized image
            data_root: Enter root data folder name
            valid_ratio: Enter valid dataset ratio
        """
        super(BaseDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.valid_ratio = valid_ratio
        self.num_classes = None
        self.num_step = None
        self.prepare_data()

    def prepare_data(self) -> None:
        train = self.dataset(root=self.data_root, train=True, download=True)
        test = self.dataset(root=self.data_root, train=False, download=True)
        self.num_classes = len(train.classes)
        self.num_step = len(train) // self.batch_size

        print('-' * 50)
        print('* {} dataset class num: {}'.format(self.dataset_name, len(train.classes)))
        print('* {} train dataset len: {}'.format(self.dataset_name, len(train)))
        print('* {} test dataset len: {}'.format(self.dataset_name, len(test)))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform)
            self.train_ds, self.valid_ds = self.split_train_valid(ds)

        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

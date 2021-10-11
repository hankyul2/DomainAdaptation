from pathlib import Path

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.data.download_dataset import download_office_31


class ImageFolderIdx(ImageFolder):
    def __getitem__(self, item):
        img, label = super(ImageFolderIdx, self).__getitem__(item)
        return img, label, item


class OFFICE31(LightningDataModule):
    def __init__(self, dataset_name: str, size: tuple, data_root: str, batch_size: int, num_workers: int, valid_ratio: float, return_idx: bool = False):
        super(OFFICE31, self).__init__()

        src, tgt = dataset_name.split('_')
        resize, crop = size[:2], size[2:]

        self.dataset_name = dataset_name
        self.src = src
        self.tgt = tgt
        self.src_root = f'{data_root}/office_31/{src}/images'
        self.tgt_root = f'{data_root}/office_31/{tgt}/images'
        self.train_dataset = ImageFolder if not return_idx else ImageFolderIdx
        self.test_dataset = ImageFolder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio

        self.num_step = None
        self.num_classes = None
        self.train_transform, self.test_transform = self.get_trasnforms(resize, crop)
        self.prepare_data()

    def prepare_data(self) -> None:
        Path("data").mkdir(exist_ok=True)
        download_office_31()
        src_ds = self.train_dataset(root=self.src_root)
        tgt_ds = self.train_dataset(root=self.tgt_root)

        self.num_classes = len(src_ds.classes)
        self.num_step = max(int(len(src_ds) * (1 - self.valid_ratio)), len(tgt_ds)) // self.batch_size

        print('-' * 50)
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} dataset number of class: {}'.format(self.tgt, self.num_classes))
        print('* {} train dataset len: {}'.format(self.src, len(src_ds) * (1 - self.valid_ratio)))
        print('* {} train dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('* {} valid dataset len: {}'.format(self.src, len(src_ds) * self.valid_ratio))
        print('* {} test dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('-' * 50)

    def setup(self, stage: str = None):
        src = self.train_dataset(self.src_root, self.train_transform)
        self.train_src_ds, self.valid_ds = self.split_train_valid(src)
        self.train_tgt_ds = self.train_dataset(self.tgt_root, self.train_transform)
        self.test_ds = self.test_dataset(self.tgt_root, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        src = DataLoader(self.train_src_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        tgt = DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return [src, tgt]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_trasnforms(self, resize, crop, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train, test

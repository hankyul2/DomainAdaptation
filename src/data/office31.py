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


class SourceOnly(LightningDataModule):
    def __init__(self, dataset_name: str, size: tuple, data_root: str, batch_size: int,
                 num_workers: int, valid_ratio: float, drop_last: bool = True):
        super(SourceOnly, self).__init__()

        data_name_list = ['amazon', 'dslr', 'webcam']
        data_name_list.remove(dataset_name)
        resize, crop = size[:2], size[2:]

        self.dataset_name = dataset_name
        self.src = dataset_name
        self.tgt1, self.tgt2 = data_name_list
        self.src_root = f'{data_root}/office_31/{self.src}/images'
        self.tgt1_root = f'{data_root}/office_31/{self.tgt1}/images'
        self.tgt2_root = f'{data_root}/office_31/{self.tgt2}/images'
        self.dataset = ImageFolder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.drop_last = drop_last

        self.num_step = None
        self.num_classes = None
        self.train_transform, self.test_transform = get_transform(resize, crop)
        self.prepare_data()

    def prepare_data(self) -> None:
        Path("data").mkdir(exist_ok=True)
        download_office_31()
        src_ds = self.dataset(root=self.src_root)
        tgt1_ds = self.dataset(root=self.tgt1_root)
        tgt2_ds = self.dataset(root=self.tgt2_root)

        self.num_classes = len(src_ds.classes)
        self.num_step = int(len(src_ds) * (1 - self.valid_ratio)) // self.batch_size

        print('-' * 50)
        print('Source Only Dataset')
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} train dataset len: {}'.format(self.src, len(src_ds) * (1 - self.valid_ratio)))
        print('* {} valid dataset len: {}'.format(self.src, len(src_ds) * self.valid_ratio))
        print('* {} test dataset len: {}'.format(self.tgt1, len(tgt1_ds)))
        print('* {} test dataset len: {}'.format(self.tgt2, len(tgt2_ds)))
        print('-' * 50)

    def setup(self, stage: str = None):
        src_ds = self.dataset(self.src_root, self.train_transform)
        self.train_src_ds, self.valid_ds = self.split_train_valid(src_ds)
        self.test1_ds = self.dataset(self.tgt1_root, transform=self.test_transform)
        self.test2_ds = self.dataset(self.tgt2_root, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_src_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test1 = DataLoader(self.test1_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test2 = DataLoader(self.test2_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return [test1, test2]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        test1 = DataLoader(self.test1_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test2 = DataLoader(self.test2_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return [test1, test2]


def get_transform(resize, crop, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
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


class DomainAdaptation(LightningDataModule):
    def __init__(self, dataset_name: str, size: tuple, data_root: str, batch_size: int, num_workers: int,
                 valid_ratio: float, num_step_mode: str = 'max', drop_last: bool = True):
        super(DomainAdaptation, self).__init__()

        src, tgt = dataset_name.split('_')
        resize, crop = size[:2], size[2:]

        self.dataset_name = dataset_name
        self.src = src
        self.tgt = tgt
        self.src_root = f'{data_root}/office_31/{src}/images'
        self.tgt_root = f'{data_root}/office_31/{tgt}/images'
        self.dataset = ImageFolder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.num_step_mode = num_step_mode
        self.drop_last = drop_last

        self.num_step = None
        self.num_classes = None
        self.train_transform, self.test_transform = get_transform(resize, crop)
        self.prepare_data()

    def prepare_data(self) -> None:
        Path("data").mkdir(exist_ok=True)
        download_office_31()
        src_ds = self.dataset(root=self.src_root)
        tgt_ds = self.dataset(root=self.tgt_root)

        self.num_classes = len(src_ds.classes)
        ds_len = max(len(src_ds), len(tgt_ds)) if self.num_step_mode == 'max' else min(len(src_ds), len(tgt_ds))
        self.num_step = ds_len // self.batch_size

        print('-' * 50)
        print('Domain Adaptation Dataset')
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} dataset number of class: {}'.format(self.tgt, self.num_classes))
        print('* {} train dataset len: {}'.format(self.src, len(src_ds)))
        print('* {} train dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('* {} test(valid) dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('-' * 50)

    def setup(self, stage: str = None):
        self.train_src_ds = self.dataset(self.src_root, self.train_transform)
        self.train_tgt_ds = self.dataset(self.tgt_root, self.train_transform)
        self.test_tgt_ds = self.dataset(self.tgt_root, transform=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        src = DataLoader(self.train_src_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last)
        tgt = DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last)
        return [src, tgt]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_tgt_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_tgt_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_tgt_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class SourceFreeDomainAdaptation(DomainAdaptation):
    def __init__(self, *args, return_idx: bool = False, **kwargs):
        super(SourceFreeDomainAdaptation, self).__init__(*args, **kwargs)
        self.train_dataset = ImageFolderIdx if return_idx else ImageFolder

    def prepare_data(self) -> None:
        Path("data").mkdir(exist_ok=True)
        download_office_31()
        src_ds = self.dataset(root=self.src_root)
        tgt_ds = self.dataset(root=self.tgt_root)

        self.num_classes = len(src_ds.classes)
        self.num_step = len(tgt_ds) // self.batch_size

        print('-' * 50)
        print('Source Free Domain Adaptation Dataset')
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} dataset number of class: {}'.format(self.tgt, self.num_classes))
        print('* {} train dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('* {} test(valid) dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('-' * 50)

    def setup(self, stage: str = None):
        self.train_tgt_ds = self.train_dataset(self.tgt_root, self.train_transform)
        self.test_tgt_ds = self.dataset(self.tgt_root, transform=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last)

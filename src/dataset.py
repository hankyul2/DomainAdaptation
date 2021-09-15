import glob
import math
import os
import random

import albumentations as A
import torch
from torch.utils.data import random_split

from PIL import Image
import numpy as np


def split_train_valid(org_ds, valid_ratio=0.1, seed=1997):
    return random_split(org_ds, [math.floor(len(org_ds) * (1 - valid_ratio)), math.ceil(len(org_ds) * valid_ratio)],
                        generator=torch.Generator().manual_seed(seed))


def get_path_label(names, name2list):
    dataset = []
    idx = 0
    for name in names:
        if len(name2list[name]) == 0:
            continue
        for item in name2list[name]:
            dataset.append((item, idx))
        idx += 1
    return dataset, idx


def get_data_list(dataset_name):
    if dataset_name == 'amazon':
        base_path = 'data/office_31/amazon/images'
    elif dataset_name == 'webcam':
        base_path = 'data/office_31/webcam/images'
    elif dataset_name == 'dslr':
        base_path = 'data/office_31/dslr/images'
    elif dataset_name == 'art':
        base_path = 'data/office_home/OfficeHomeDataset_10072016/Art'
    elif dataset_name == 'clip_art':
        base_path = 'data/office_home/OfficeHomeDataset_10072016/Clipart'
    elif dataset_name == 'product':
        base_path = 'data/office_home/OfficeHomeDataset_10072016/Product'
    elif dataset_name == 'real_world':
        base_path = 'data/office_home/OfficeHomeDataset_10072016/\'Real World\''
    class_names = [os.path.basename(path) for path in glob.glob(os.path.join(base_path, '*'))]
    class_name2path = {name: glob.glob(os.path.join(base_path, name, '*')) for name in class_names}
    ds, nclass = get_path_label(class_names, class_name2path)
    return ds, nclass


def get_dataset(src, tgt):
    src_data_list, src_nclass = get_data_list(src)
    tgt_data_list, tgt_nclass = get_data_list(tgt)

    transforms_train = A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(256, 256, p=1),
        A.RandomSizedCrop([200, 256], 224, 224),
        A.HorizontalFlip(),
        A.Normalize()
    ])

    transforms_test = A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224, p=1),
        A.Normalize()
    ])

    train_src_list, valid_src_list = split_train_valid(src_data_list)
    fake_size = max(len(train_src_list), len(tgt_data_list))
    train_src_ds = MyDataset(train_src_list, src_nclass, fake_size, transforms=transforms_train)
    train_tgt_ds = MyDataset(tgt_data_list, tgt_nclass, fake_size, transforms=transforms_train)
    valid_ds = MyDataset(valid_src_list, transforms=transforms_test)
    test_ds = MyDataset(tgt_data_list, transforms=transforms_test)

    print('{} dataset number of class: {}'.format(src, src_nclass))
    print('{} dataset number of class: {}'.format(tgt, tgt_nclass))
    print('{} train src dataset len: {}'.format(src, len(train_src_list)))
    print('{} train tgt dataset len: {}'.format(tgt, len(train_tgt_ds)))
    print('{} valid src dataset len: {}'.format(src, len(valid_ds)))
    print('{} test tgt dataset len: {}'.format(src, len(test_ds)))

    return train_src_ds, train_tgt_ds, valid_ds, test_ds


def convert_to_dataloader(datasets, batch_size, num_workers, shuffle=True, drop_last=True, sampler_fn=None):
    return [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(shuffle if not sampler_fn else False), num_workers=num_workers,
                                        drop_last=drop_last, sampler=(sampler_fn(ds) if sampler_fn else None)) for ds in datasets]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_num=None, fake_size=None, transforms=None):
        super().__init__()
        self.df = df
        self.size = len(self.df)
        self.fake_size = fake_size if fake_size else self.size
        self.transforms = transforms
        self.class_num = class_num
        self.limit = self.fake_size - (self.fake_size % self.size)

    def __len__(self):
        return self.fake_size

    def __getitem__(self, idx):
        img, label = self.df[self.compute_idx(idx)]
        img = Image.open(img).convert('RGB')

        if self.transforms is not None:
            img = np.transpose(self.transforms(image=np.array(img))['image'], (2, 0, 1))

        return img, label

    def compute_idx(self, idx):
        """This method is made for uniform distribution of smaller dataset"""
        return idx % self.size if idx < self.limit else random.randint(0, self.size-1)

import glob
import os
import albumentations as A
import torch

from PIL import Image
import numpy as np

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
    wecam_names = [os.path.basename(path) for path in glob.glob(os.path.join(base_path, '*'))]
    webcam_name2path = {name: glob.glob(os.path.join(base_path, name, '*')) for name in wecam_names}
    ds, webcam_nclass = get_path_label(wecam_names, webcam_name2path)
    return ds, webcam_nclass


def get_dataset(src, tgt):
    src_ds, src_nclass = get_data_list(src)
    tgt_ds, tgt_nclass = get_data_list(tgt)

    transforms = A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(256, 256, p=1),
        A.RandomSizedCrop([200, 256], 224, 224),
        A.HorizontalFlip(),
        A.Normalize()
    ])

    fake_size = max(len(src_ds), len(tgt_ds))
    src_dataset = MyDataset(src_ds, fake_size, transforms=transforms)
    tgt_dataset = MyDataset(tgt_ds, fake_size, transforms=transforms)
    test_dataset = MyDataset(tgt_ds, transforms=transforms)

    print('{} dataset number of class: {}'.format(src, src_nclass))
    print('{} dataset number of class: {}'.format(tgt, tgt_nclass))
    print('{} dataset len: {}'.format(src, len(src_ds)))
    print('{} dataset len: {}'.format(tgt, len(tgt_ds)))

    return src_dataset, tgt_dataset, test_dataset


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, fake_size=None, transforms=None):
        super().__init__()
        self.df = df
        self.size = len(self.df)
        self.fake_size = fake_size
        self.transforms = transforms

    def __len__(self):
        return self.fake_size if self.fake_size is not None else self.size

    def __getitem__(self, idx):
        img, label = self.df[idx % self.size]
        img = Image.open(img).convert('RGB')

        if self.transforms is not None:
            img = np.transpose(self.transforms(image=np.array(img))['image'], (2, 0, 1))

        return img, label
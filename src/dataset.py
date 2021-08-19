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

    transforms_train_valid = A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(256, 256, p=1),
        A.RandomSizedCrop([200, 256], 224, 224),
        A.HorizontalFlip(),
        A.Normalize()
    ])

    transforms_test = A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(256, 256, p=1),
        A.RandomSizedCrop([200, 256], 224, 224),
        A.HorizontalFlip(),
        A.Normalize()
    ])

    fake_size = max(len(src_data_list), len(tgt_data_list))
    src_dataset = MyDataset(src_data_list, src_nclass, fake_size, transforms=transforms_train_valid)
    tgt_dataset = MyDataset(tgt_data_list, tgt_nclass, fake_size, transforms=transforms_train_valid)
    valid_dataset = MyDataset(tgt_data_list, transforms=transforms_train_valid)
    test_dataset = MyDataset(tgt_data_list, transforms=transforms_test)

    print('{} dataset number of class: {}'.format(src, src_nclass))
    print('{} dataset number of class: {}'.format(tgt, tgt_nclass))
    print('{} dataset len: {}'.format(src, len(src_data_list)))
    print('{} dataset len: {}'.format(tgt, len(tgt_data_list)))

    return src_dataset, tgt_dataset, valid_dataset, test_dataset


def convert_to_dataloader(datasets, batch_size, num_workers, train=True):
    return [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                                        drop_last=True) for ds in datasets]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_num=None, fake_size=None, transforms=None):
        super().__init__()
        self.df = df
        self.size = len(self.df)
        self.fake_size = fake_size
        self.transforms = transforms
        self.class_num = class_num

    def __len__(self):
        return self.fake_size if self.fake_size else self.size

    def __getitem__(self, idx):
        img, label = self.df[idx % self.size]
        img = Image.open(img).convert('RGB')

        if self.transforms is not None:
            img = np.transpose(self.transforms(image=np.array(img))['image'], (2, 0, 1))

        return img, label
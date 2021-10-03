from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from src.data.base_data_module import BaseDataModule


class CIFAR(BaseDataModule):
    def __init__(self, dataset_name: str, size: tuple, **kwargs):
        if dataset_name == 'cifar10':
            dataset, mean, std = CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_name == 'cifar100':
            dataset, mean, std = CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

        train_transform, test_transform = self.get_trasnforms(mean, std, size)
        super(CIFAR, self).__init__(dataset_name, dataset, train_transform, test_transform, **kwargs)

    def get_trasnforms(self, mean, std, size):
        train = transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train, test

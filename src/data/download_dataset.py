import os
import pathlib
import zipfile
import gdown
from torchvision.datasets import MNIST, SVHN, USPS


def download_office_home():
    if os.path.exists('data/office_home.zip'):
        print('office-home data already downloaded')
        return
    url = 'https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg'
    store_path = 'data/office_home.zip'
    gdown.download(url, store_path, quiet=False)
    unzip('data/office_home.zip', 'data/office_home')


def download_office_31():
    if os.path.exists('data/office_31.zip'):
        print('office-31 data already downloaded')
        return
    url = 'https://drive.google.com/uc?id=1x-qGoVeIpmX_92UW-aEl1Xy9WftUOvMT'
    store_path = 'data/office_31.zip'
    gdown.download(url, store_path, quiet=False)
    unzip('data/office_31.zip', 'data/office_31')


def download_digits():
    MNIST(root='data/', download=True)
    SVHN(root='data/', download=True)
    USPS(root='data/', download=True)


def unzip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


if __name__ == '__main__':
    pathlib.Path("data").mkdir(exist_ok=True)
    download_office_home()
    download_office_31()
    download_digits()

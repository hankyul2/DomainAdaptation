import os
import subprocess
from pathlib import Path

from torch.hub import load_state_dict_from_url

import numpy as np


model_urls = {
    # ResNet
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

    # MobileNetV2
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',

    # Se ResNet
    'seresnet18': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet18-4bb0ce65.pth',
    'seresnet34': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet34-a4004e63.pth',
    'seresnet50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth',
    'seresnet101': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet101-7e38fcc6.pth',
    'seresnet152': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet152-d17c99b7.pth',
    'seresnext50_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',

    # ViT
    'vit_base_patch16_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
    'vit_base_patch32_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz',
    'vit_large_patch16_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz',
    'vit_large_patch32_224': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz',

    # Hybrid (resnet50 + ViT)
    'r50_vit_base_patch16_224': 'https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz',
    'r50_vit_large_patch32_224': 'https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-L_32.npz',
}


def load_from_zoo(model, model_name, pretrained_path='pretrained/official'):
    model_name = change_384_224(model_name)
    Path(os.path.join(pretrained_path, model_name)).mkdir(parents=True, exist_ok=True)
    if model_urls[model_name].endswith('pth'):
        state_dict = load_state_dict_from_url(url=model_urls[model_name],
                                              model_dir=os.path.join(pretrained_path, model_name),
                                              progress=True, map_location='cpu')
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        state_dict.pop('classifier.weight', None)
        state_dict.pop('classifier.bias', None)

        model.load_state_dict(state_dict, strict=False)
    elif model_urls[model_name].endswith('npz'):
        npz = load_npz_from_url(url=model_urls[model_name],
                                file_name=os.path.join(pretrained_path, model_name, os.path.basename(model_urls[model_name])))
        model.load_npz(npz)


def change_384_224(model_name):
    model_name = model_name.replace('384', '224')
    return model_name


def load_npz_from_url(url, file_name):
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
    return np.load(file_name)


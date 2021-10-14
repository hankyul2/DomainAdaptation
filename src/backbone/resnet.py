import re
from typing import Type, Union

import torch
from einops import rearrange
from torch import nn

from src.backbone.layers.conv_block import BasicBlock, BottleNeck, conv1x1, resnet_normal_init, resnet_zero_init, \
    PreActBasicBlock, PreActBottleNeck
from src.backbone.utils import load_from_zoo


class ResNet(nn.Module):
    def __init__(self,
                 nblock: list,
                 block: Type[Union[BasicBlock, PreActBasicBlock, PreActBottleNeck, BottleNeck]],
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 channels: list = [64, 128, 256, 512],
                 strides=[1, 2, 2, 2],
                 groups=1,
                 base_width=64) -> None:
        super(ResNet, self).__init__()
        self.groups = groups
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]
        self.out_channels = channels[-1] * block.factor

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=2, padding=(3, 3), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.layers = [self.make_layer(block=block, nblock=nblock[i], channels=channels[i], stride=strides[i]) for i in range(len(nblock))]
        self.register_layer()

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def make_layer(self, block: Type[Union[BasicBlock, PreActBasicBlock, PreActBottleNeck, BottleNeck]], nblock: int, channels: int, stride: int) -> nn.Sequential:
        layers = []
        downsample = None
        if self.in_channels != channels * block.factor:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.factor, stride=stride),
                self.norm_layer(channels * block.factor)
            )
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                self.in_channels = channels * block.factor
            layers.append(block(in_channels=self.in_channels, out_channels=channels,
                                stride=stride, norm_layer=self.norm_layer, downsample=downsample,
                                groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.features(x)
        return self.flatten(self.avgpool(x))

    def load_npz(self, npz):
        name_convertor = [
            ('downsample.0', 'conv_proj'), ('downsample.1', 'gn_proj'),
            ('^conv1.weight', 'conv_root/kernel'), ('^bn1', 'bn_root'),
            ('bn', 'gn'), ('layer', 'block'), ('conv(.*)weight', 'conv\\1kernel'),
            ('\\.(\\d)\\.', lambda x: f'.unit{int(x.group(1))+1}.'),
            ('weight', 'scale'),  ('\\.', '/'),
        ]
        for name, param in self.named_parameters():
            if 'fc' in name:
                continue
            for pattern, sub in name_convertor:
                name = re.sub(pattern, sub, name)
            param.data.copy_(npz_dim_convertor(name, npz.get(name)))


def npz_dim_convertor(name, weight):
    weight = torch.from_numpy(weight)
    if 'kernel' in name:
        weight = rearrange(weight, 'h w in_c out_c -> out_c in_c h w')
    elif 'scale' in name or 'bias' in name:
        weight = weight.squeeze()
    return weight


def get_resnet(model_name: str, zero_init_residual=False, pretrained=False, **kwargs) -> nn.Module:
    if model_name == 'resnet18':
        model = ResNet(nblock=[2, 2, 2, 2], block=BasicBlock)
    elif model_name == 'resnet34':
        model = ResNet(nblock=[3, 4, 6, 3], block=BasicBlock)
    elif model_name == 'resnet50':
        model = ResNet(nblock=[3, 4, 6, 3], block=BottleNeck)
    elif model_name == 'resnet101':
        model = ResNet(nblock=[3, 4, 23, 3], block=BottleNeck)
    elif model_name == 'resnet152':
        model = ResNet(nblock=[3, 8, 36, 3], block=BottleNeck)
    elif model_name == 'resnext50_32x4d':
        model = ResNet(nblock=[3, 8, 36, 3], block=BottleNeck, groups=32, base_width=4)
    elif model_name == 'wide_resnet50_2':
        model = ResNet(nblock=[3, 8, 36, 3], block=BottleNeck, base_width=128)

    resnet_normal_init(model)
    resnet_zero_init(model, zero_init_residual)

    if pretrained:
        load_from_zoo(model, model_name)

    return model



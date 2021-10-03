from typing import Union, Type

from torch import nn
from src.backbone.layers.conv_block import conv1x1, resnet_normal_init, resnet_zero_init, BasicBlock, BottleNeck


class ResNet32(nn.Module):

    def __init__(self,
                 block_list: tuple = (3, 3, 3),
                 channels: tuple = (16, 32, 64),
                 norm_layer: Type[Union[nn.BatchNorm2d, nn.GroupNorm]] = nn.BatchNorm2d,
                 block: Type[Union[BasicBlock, BottleNeck]] = BasicBlock,
                 groups: int = 1,
                 base_width: int = 64):
        super(ResNet32, self).__init__()
        self.groups = groups
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.block = block
        self.in_channels = channels[0]
        self.out_channels = channels[-1]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=1, padding=(1, 1), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.layers = [self.make_layer(block=self.block, nblock=block_list[i], channels=channels[i]) for i in
                       range(len(block_list))]
        self.register_layer()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            x = layer(x)
        return self.flatten(self.avgpool(x))

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def make_layer(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: int, channels: int) -> nn.Sequential:
        layers = []
        downsample = None
        stride = 1
        if self.in_channels != channels * block.factor:
            stride = 2
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.factor, stride=stride),
                nn.BatchNorm2d(channels * block.factor)
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


def get_resnet32(model_name, zero_init_residual=False, **kwargs):
    if model_name == 'resnet32_20':
        model = ResNet32(block_list=(3, 3, 3))
    elif model_name == 'resnet32_110':
        model = ResNet32(block_list=(18, 18, 18))

    resnet_normal_init(model)
    resnet_zero_init(model, zero_init_residual)

    return model
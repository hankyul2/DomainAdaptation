"""Squeeze-Excitation Net (SENet, 2018)
This implementation follows timm repo, https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/senet.py
I only add SE ResNet version, which means only on SEBottleNeck exists here.
"""
from collections import OrderedDict

from torch import nn

from src.backbone.layers.conv_block import SEBasicBlock, SEBottleNeck
from src.backbone.resnet import ResNet
from src.backbone.utils import load_from_zoo


class SeResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super(SeResNet, self).__init__(*args, **kwargs)
        self.layer0 = nn.Sequential(OrderedDict([('conv1', self.conv1), ('bn1', self.bn1), ('relu1', self.relu), ('maxpool', self.maxpool)]))
        self.layers.insert(0, self.layer0)
        del self.conv1
        del self.bn1
        del self.relu

    def features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_seresnet(model_name: str, pretrained=False, **kwargs) -> nn.Module:
    if model_name == 'seresnet18':
        model = SeResNet(nblock=[2, 2, 2, 2], block=SEBasicBlock)
    elif model_name == 'seresnet34':
        model = SeResNet(nblock=[3, 4, 6, 3], block=SEBasicBlock)
    elif model_name == 'seresnet50':
        model = SeResNet(nblock=[3, 4, 6, 3], block=SEBottleNeck)
    elif model_name == 'seresnet101':
        model = SeResNet(nblock=[3, 4, 23, 3], block=SEBottleNeck)
    elif model_name == 'seresnet152':
        model = SeResNet(nblock=[3, 8, 36, 3], block=SEBottleNeck)
    elif model_name == 'seresnext50_32x4d':
        model = SeResNet(nblock=[3, 8, 36, 3], block=SEBottleNeck, groups=32, base_width=4)
    else:
        raise AssertionError("No model like that in SE ResNet model")

    if pretrained:
        load_from_zoo(model, model_name)

    return model
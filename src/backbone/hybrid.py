from functools import partial

import torch
from torch import nn

from src.backbone.layers.conv_block import BottleNeck, StdConv
from src.backbone.resnet import ResNet
from src.backbone.vit import build_vit
from src.backbone.utils import load_from_zoo


class Hybrid(nn.Module):
    def __init__(self, cnn, vit):
        super(Hybrid, self).__init__()
        self.cnn = cnn
        self.vit = vit
        self.out_channels = vit.out_channels

    def forward(self, x):
        x = self.cnn.features(x)
        return self.vit(x)

    def load_npz(self, npz):
        self.cnn.load_npz(npz)
        self.vit.load_npz(npz)


def get_hybrid(model_name, pretrained=False, pre_logits=True, dropout=0.1, **kwargs):
    if 'vit_base' in model_name:
        num_layer, d_model, h, d_ff, N = [3, 4, 9], 768, 12, 3072, 12
    elif 'vit_large' in model_name:
        num_layer, d_model, h, d_ff, N = [3, 4, 6, 3], 1024, 16, 4096, 24

    cnn = ResNet(nblock=num_layer, block=BottleNeck, norm_layer=partial(nn.GroupNorm, 32), conv=StdConv)
    feature_dim, feature_size = get_feature_map_info(cnn, model_name)
    vit = build_vit(patch_size=(1, 1), img_size=feature_size, in_channel=feature_dim, d_model=d_model,
                    h=h, d_ff=d_ff, N=N, pre_logits=pre_logits, dropout=dropout)
    hybrid = Hybrid(cnn=cnn, vit=vit)

    if pretrained:
        load_from_zoo(hybrid, model_name)

    return hybrid


@torch.no_grad()
def get_feature_map_info(cnn, model_name):
    if '224' in model_name:
        img_size = (1, 3, 224, 224)
    elif '384' in model_name:
        img_size = (1, 3, 384, 384)
    _, c, h, w = map(int, cnn.features(torch.rand(img_size)).shape)
    return c, (h, w)
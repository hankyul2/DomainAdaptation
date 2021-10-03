from einops import rearrange, repeat
from torch import nn
import torch


def is_pair(img_size):
    return img_size if isinstance(img_size, tuple) else (img_size, img_size)


def get_patch_num_and_dim(img_size, patch_size, in_channel=3):
    i_h, i_w = is_pair(img_size)
    p_h, p_w = is_pair(patch_size)
    assert i_h % p_h == 0 and i_w % p_w == 0

    patch_num = (i_h // p_h) * (i_w // p_w)
    patch_dim = p_h * p_w * in_channel
    return patch_num, patch_dim


class BasicLinearProjection(nn.Module):
    def __init__(self, d_model=512, patch_size=(16, 16), in_channel=3):
        super(BasicLinearProjection, self).__init__()
        p_h, p_w = is_pair(patch_size)
        self.img2patch = lambda x: rearrange(x, 'b c (p h) (q w) -> b (p q) (h w c)', h=p_h, w=p_w)
        self.linear_projection = nn.Linear(p_h*p_w*in_channel, d_model)

    def forward(self, x):
        x = self.img2patch(x)
        return self.linear_projection(x)


class ConvLinearProjection(nn.Module):
    def __init__(self, d_model=512, patch_size=(16, 16), in_channel=3):
        super(ConvLinearProjection, self).__init__()
        p_h, p_w = is_pair(patch_size)
        self.linear_projection = nn.Conv2d(in_channel, d_model, p_h, p_h)
        self.img2patch = lambda x: rearrange(x, 'b e p q -> b (p q) e')

    def forward(self, x):
        x = self.linear_projection(x)
        return self.img2patch(x)


class TokenLayer(nn.Module):
    def __init__(self, d_model):
        super(TokenLayer, self).__init__()
        self.cls_token = nn.Parameter(torch.rand((1, 1, d_model,)))
        self.pad_cls_token = lambda x: torch.cat([repeat(self.cls_token, '1 1 d -> b 1 d', b=x.size(0)), x], dim=1)

    def forward(self, x):
        return self.pad_cls_token(x)


class PositionalEncoding(nn.Module):
    def __init__(self, patch_num, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.rand(patch_num+1, d_model))
        self.add_positional_encoding = lambda x: x + self.pe[:x.size(1)].unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.add_positional_encoding(x)
        return self.dropout(x)







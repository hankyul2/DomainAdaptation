import torch
from torch import nn
import torch.nn.functional as F

from src.system.dann import DANN


class Centroid(nn.Module):
    def __init__(self, num_classes:int, embed_dim: int, theta: float = 0.7):
        super(Centroid, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.theta = theta
        self.register_buffer('eye', torch.eye(self.num_classes))
        self.register_buffer('centroid', torch.zeros(num_classes, embed_dim))

    def forward(self, embed, y):
        weight = self.eye @ self.eye[y].t()
        cur_centroid = weight @ embed / (weight.sum(dim=1, keepdim=True) + 1e-8)
        return self.update_centroid(cur_centroid)

    def update_centroid(self, cur_centroid):
        centroid = self.theta * self.centroid + (1 - self.theta) * cur_centroid
        self.centroid.data = centroid.data
        return centroid


class MSTN(DANN):
    def __init__(self, *args, **kwargs):
        super(MSTN, self).__init__(*args, **kwargs)
        self.src_centroid = Centroid(self.num_classes, kwargs['embed_dim'])
        self.tgt_centroid = Centroid(self.num_classes, kwargs['embed_dim'])

    def training_step(self, batch, batch_idx, optimizer_idx=None, child_compute_already=None):
        (x_s, y_s), (x_t, y_t) = batch
        embed_s, y_hat_s = self.get_feature(x_s, 'src')
        embed_t, y_hat_t = self.get_feature(x_t, 'tgt')
        loss_centroid = F.mse_loss(self.src_centroid(embed_s, y_s), self.tgt_centroid(embed_t, y_hat_t.argmax(dim=1)))
        loss = super(MSTN, self).training_step(batch, batch_idx, optimizer_idx, ((embed_s, y_hat_s), (embed_t, y_hat_t)))
        return loss + loss_centroid * self.get_alpha()
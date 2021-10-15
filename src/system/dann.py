import numpy as np
import torch
from torch import nn
from pytorch_lightning.utilities.cli import instantiate_class

from src.common_module import DomainClassifier
from src.system.source_only import DABase


class DANN(DABase):
    def __init__(self, *args, hidden_dim: int = 1024, gamma: int = 10, **kwargs):
        super(DANN, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.dc = DomainClassifier(kwargs['embed_dim'], hidden_dim)
        self.criterion_dc = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, optimizer_idx=None, child_compute_already=None):
        (x_s, y_s), (x_t, y_t) = batch
        if child_compute_already:
            (embed_s, y_hat_s), (embed_t, y_hat_t) = child_compute_already
        else:
            embed_s, y_hat_s = self.get_feature(x_s)
            embed_t, y_hat_t = self.get_feature(x_t)

        loss_dc = self.compute_dc_loss(embed_s, embed_t, y_hat_s, y_hat_t)
        loss_cls = self.criterion(y_hat_s, y_s)
        loss = loss_cls+loss_dc

        metric = self.train_metric(y_hat_s, y_s)
        self.log_dict({f'train/loss': loss})
        self.log_dict(metric)
        return loss

    def compute_dc_loss(self, embed_s, embed_t, y_hat_s, y_hat_t):
        y_hat_dc = self.dc(torch.cat([embed_s, embed_t]), self.get_alpha())
        y_dc = torch.cat([torch.zeros_like(y_hat_s[:, 0]), torch.ones_like(y_hat_t[:, 0])]).long()
        loss_dc = self.criterion_dc(y_hat_dc, y_dc)
        return loss_dc

    def get_feature(self, x, domain=None):
        feature = self.backbone(x)
        embed = self.bottleneck(feature)
        y_hat = self.fc(embed)
        return embed, y_hat

    def get_alpha(self):
        return 2. / (1. + np.exp(-self.gamma * self.global_step / (self.num_step * self.max_epochs))) - 1

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.bottleneck.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.dc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
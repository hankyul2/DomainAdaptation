import os
import copy
from pathlib import Path
from collections import OrderedDict

import torch
from torch import nn

from pytorch_lightning.utilities.cli import instantiate_class

from src.common_module import LabelSmoothing
from src.system.base import BaseVisionSystem


class DABase(BaseVisionSystem):
    def __init__(self, *args, embed_dim: int = 256, dropout: float = 0.1, **kwargs):
        super(DABase, self).__init__(*args, **kwargs)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone.out_channels, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(embed_dim, kwargs['num_classes'])
        self.criterion = LabelSmoothing()

    def forward(self, x):
        return self.fc(self.backbone(self.backbone(x)))

    def compute_loss(self, x, y):
        return self.compute_loss_eval(x, y)

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.bottleneck(self.backbone(x)))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.bottleneck.parameters()},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join('pretrained', self.__class__.__name__ + '_' + self.backbone.__class__.__name__, self.trainer.datamodule.dataset_name + '.ckpt')
        Path(os.path.dirname(save_path)).mkdir(exist_ok=True, parents=True)
        weight = nn.Sequential(OrderedDict([('backbone', self.backbone), ('bottleneck', self.bottleneck), ('fc', self.fc)])).state_dict()
        with open(save_path, 'wb') as f:
            torch.save(weight, f)




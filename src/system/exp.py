import os

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import instantiate_class

from src.common_module import entropy, divergence, LabelSmoothing, GRL
from src.system.cdan import CDAN_E
from src.system.mstn import MSTN
from src.system.shot import SHOT
from src.system.source_only import DABase


class SHOT_CDAN(SHOT):
    def on_fit_start(self) -> None:
        weight_path = os.path.join(self.source_only_path, self.trainer.datamodule.dataset_name + '.ckpt')
        self.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)


class CDAN_E_MSTN(CDAN_E, MSTN):
    pass


class PSEUDO_MIXUP_RATIO_CDAN(DABase):
    def __init__(self, *args, source_only_path: str = None, **kwargs):
        super(PSEUDO_MIXUP_RATIO_CDAN, self).__init__(*args, **kwargs)
        self.source_only_path = source_only_path
        self.criterion = LabelSmoothing(reduction='none')
        self.ratio_cls = nn.Sequential(
            nn.Linear(kwargs['embed_dim'], 1024), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
        )

    def on_fit_start(self) -> None:
        weight_path = os.path.join(self.source_only_path, self.trainer.datamodule.dataset_name + '.ckpt')
        self.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        (x_s, y_s), (x_t, y_t) = batch

        ratio = torch.rand(x_s.size(0)).to(self.device)
        embed_s, y_hat_s = self.get_feature(x_s)
        embed_t, y_hat_t = self.get_feature(x_t)
        embed_r, y_hat_r = self.get_feature(x_s * ratio.view(-1, 1, 1, 1) + x_t * (1 - ratio.view(-1, 1, 1, 1)))

        prob, pred = F.softmax(y_hat_t, dim=1).max(dim=1)
        mask = torch.nonzero(torch.gt(prob, 0.95), as_tuple=True)[0]

        loss_cls = self.criterion(y_hat_s, y_s).mean() + self.criterion(y_hat_t[mask], pred[mask]).mean()
        loss_mix = (self.criterion(y_hat_r[mask], y_s[mask]).sum(dim=1) * ratio[mask] +
                    self.criterion(y_hat_r[mask],pred[mask]).sum(dim=1) * (1 - ratio[mask])).mean()
        loss_e = self.im_loss(y_hat_s) + self.im_loss(y_hat_t) + self.im_loss(y_hat_r)
        loss = loss_cls + loss_mix + loss_e

        metric = self.train_metric(y_hat_s, y_s)
        self.log_dict({'train/loss_mix': loss_mix, 'train/loss_e': loss_e})
        self.log_dict({f'train/loss': loss})
        self.log_dict(metric)
        return loss

    def im_loss(self, x):
        p = F.softmax(x, dim=1)
        return entropy(p).mean() - divergence(p)

    def get_feature(self, x, domain=None):
        feature = self.backbone(x)
        embed = self.bottleneck(feature)
        y_hat = self.fc(embed)
        return embed, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters()},
            {'params': self.bottleneck.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.ratio_cls.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 10},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class RatioClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        x = self.layer(x)
        return x


class RATIO_CDAN_E(CDAN_E):
    def __init__(self, *args, **kwargs):
        super(RATIO_CDAN_E, self).__init__(*args, **kwargs)
        self.rc = RatioClassifier(kwargs['embed_dim'], kwargs['hidden_dim'])

    def training_step(self, batch, batch_idx, optimizer_idx=None, child_compute_already=None):
        (x_s, y_s), (x_t, y_t) = batch
        ratio = torch.rand(x_s.size(0)).to(self.device)
        embed_r, y_hat_r = self.get_feature(x_s * ratio.view(-1, 1, 1, 1) + x_t * (1 - ratio.view(-1, 1, 1, 1)))
        loss_ratio = F.mse_loss(self.rc(embed_r, self.get_alpha()), 1 - ratio)

        return loss_ratio + super(RATIO_CDAN_E, self).training_step(batch, batch_idx, optimizer_idx, None)

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.bottleneck.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.dc.parameters()},
            {'params': self.rc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

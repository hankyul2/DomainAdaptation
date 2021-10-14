import copy

import torch
from torch import nn
import torch.nn.functional as F

from src.system.shot import SHOT


class TransDA(SHOT):
    def on_fit_start(self) -> None:
        super(TransDA, self).on_fit_start()
        self.backbone_t = copy.deepcopy(self.backbone)
        self.bottleneck_t = copy.deepcopy(self.bottleneck)
        self.backbone_t.requires_grad_(False)
        self.bottleneck_t.requires_grad_(False)

    def on_train_epoch_start(self) -> None:
        self.make_pseudo_label(nn.Sequential(self.backbone_t, self.bottleneck_t), self.fc)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        loss, y_hat = self.compute_loss(*batch) if mode == 'train' else self.compute_loss_eval(*batch)
        metric = metric(y_hat, batch[1])
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx)
        self.log_dict(metric, add_dataloader_idx=add_dataloader_idx)
        return loss

    def compute_loss(self, x, y, z):
        cls_im_loss, y_hat = super(TransDA, self).compute_loss(x, y)
        kd_loss = (-F.log_softmax(y_hat, dim=1) * F.softmax(self.pseudo_logit[z], dim=1)).mean()
        return cls_im_loss + kd_loss, y_hat

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        self.update_teacher_model()

    def update_teacher_model(self):
        self.ema_update(self.backbone, self.backbone_t)
        self.ema_update(self.bottleneck, self.bottleneck_t)

    def ema_update(self, model_s, model_t, m=0.001):
        with torch.no_grad():
            for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
                param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)





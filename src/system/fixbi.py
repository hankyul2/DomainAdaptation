import os
import copy
from functools import partial
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_lightning.utilities.cli import instantiate_class
from torchmetrics import Accuracy, MetricCollection

from src.system.source_only import DABase


def mixup(x, y, ratio):
    return x * ratio + y * (1 - ratio)


def fix_mixup_loss(model, X_s, X_t, y_s, y_t, ratio=0.7):
    y_m = model(mixup(X_s, X_t, ratio))
    return mixup(F.cross_entropy(y_m, y_s), F.cross_entropy(y_m, y_t), ratio)


def self_penalty_loss(mask_sdm, mask_tdm, y_hat_sdm, y_hat_tdm, y_t_sdm, y_t_tdm, sdt, tdt):
    # Todo: There are problem with self_penalty_loss
    return 0
    # return F.nll_loss(torch.log(1 - F.softmax(y_hat_sdm[mask_sdm]/sdt, dim=1) + 1e-8), y_t_sdm[mask_sdm]) + \
    #         F.nll_loss(torch.log(1 - F.softmax(y_hat_tdm[mask_tdm]/tdt, dim=1) + 1e-8), y_t_tdm[mask_tdm])


def bidirectional_loss(mask_sdm, mask_tdm, y_hat_sdm, y_hat_tdm, y_t_sdm, y_t_tdm):
    return F.cross_entropy(y_hat_sdm[mask_tdm], y_t_tdm[mask_tdm]) + \
            F.cross_entropy(y_hat_tdm[mask_sdm], y_t_sdm[mask_sdm])


def get_label_and_mask(y_hat, compare_fn=torch.lt):
    prob, pseudo_label = F.softmax(y_hat, dim=1).max(dim=1)
    threshold = prob.mean() - 2 * prob.std()
    mask = torch.nonzero(compare_fn(prob, threshold), as_tuple=True)[0]
    return pseudo_label, mask, threshold


def confidence_loss(y_hat_s, y_hat_t, loss_fn, compare_fn):
    pseudo_s, mask_s, threshold_s = get_label_and_mask(y_hat_s, compare_fn)
    pseudo_t, mask_t, threshold_t = get_label_and_mask(y_hat_t, compare_fn)
    mask_len = min(len(mask_s), len(mask_t))
    if mask_len:
        loss_cl = loss_fn(mask_s[:mask_len], mask_t[:mask_len], y_hat_s, y_hat_t, pseudo_s.detach(), pseudo_t.detach())
    else:
        loss_cl = 0
    return loss_cl, threshold_s, threshold_t


def consistency_loss(SDM, TDM, X_s, X_t, ratio=0.5):
    X_m = mixup(X_s, X_t, ratio)
    return F.mse_loss(SDM(X_m), TDM(X_m))


class FixBiLoss(nn.Module):
    def __init__(self, sdl: float = 0.7, tdl: float = 0.3, cdl: float = 0.5,
                 sdt_init: float = 5.0, tdt_init: float = 5.0, warmup: int = 100):
        super(FixBiLoss, self).__init__()
        self.cdl = cdl
        self.tdl = tdl
        self.sdl = sdl
        self.sdt = nn.Parameter(torch.tensor(sdt_init))
        self.tdt = nn.Parameter(torch.tensor(tdt_init))
        self.warmup = warmup

    def forward(self, SDM, TDM, X_s, y_s, X_t, epoch):
        y_hat_sdm, y_hat_tdm = SDM(X_t), TDM(X_t)

        loss_fm = fix_mixup_loss(SDM, X_s, X_t, y_s, y_hat_sdm.argmax(dim=1), self.sdl) + \
                    fix_mixup_loss(TDM, X_s, X_t, y_s, y_hat_tdm.argmax(dim=1), self.tdl)

        if epoch < self.warmup:
            loss_fn, compare_fn = partial(self_penalty_loss, sdt=self.sdt, tdt=self.tdt), torch.lt
            loss_cl, threshold_sdm, threshold_tdm = confidence_loss(y_hat_sdm, y_hat_tdm, loss_fn, compare_fn)
            loss_cr = 0
        else:
            loss_fn, compare_fn = bidirectional_loss, torch.gt
            loss_cl, threshold_sdm, threshold_tdm = confidence_loss(y_hat_sdm, y_hat_tdm, loss_fn, compare_fn)
            loss_cr = consistency_loss(SDM, TDM, X_s, X_t, self.cdl)

        return loss_fm, loss_cl, loss_cr, threshold_sdm, threshold_tdm


class MyMetric(MetricCollection):
    @torch.jit.unused
    def forward(self, *args, **kwargs):
        return {k: m(*arg) for arg, (k, m) in zip(args, self.items())}


class FixBi(DABase):
    def __init__(self, *args, sd_lambda: float = 0.7, warmup: int = 100, uda_path: str = 'pretrained/dann', **kwargs):
        super(FixBi, self).__init__(*args, **kwargs)
        c = copy.deepcopy
        self.uda_path = uda_path
        self.sd_model = nn.Sequential(OrderedDict([('backbone', c(self.backbone)), ('bottleneck', c(self.bottleneck)), ('fc', c(self.fc))]))
        self.td_model = nn.Sequential(OrderedDict([('backbone', c(self.backbone)), ('bottleneck', c(self.bottleneck)), ('fc', c(self.fc))]))
        self.loss_fn = FixBiLoss(sd_lambda, 1-sd_lambda, 0.5, 5.0, 5.0, warmup)

        metric = MyMetric({"sd@1": Accuracy(top_k=1), "td@1": Accuracy(top_k=1), "top@1": Accuracy(top_k=1)})
        self.valid_metric = metric.clone(prefix='valid/')
        self.test_metric = metric.clone(prefix='test/')

    def on_fit_start(self) -> None:
        weight_path = os.path.join(self.uda_path, self.trainer.datamodule.dataset_name+'.ckpt')
        weight = torch.load(weight_path, map_location='cpu')
        self.sd_model.load_state_dict(weight, strict=False)
        self.td_model.load_state_dict(weight, strict=False)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        (src_x, src_y), (tgt_x, tgt_y) = batch
        loss_fm, loss_cl, loss_cr, threshold_sdm, threshold_tdm = \
            self.loss_fn(self.sd_model, self.td_model, src_x, src_y, tgt_x, self.current_epoch)
        self.log_dict({
            'train/loss_fm': loss_fm, 'train/loss_cl': loss_cl, 'train/loss_cr': loss_cr,
            'train/threshold_sdm': threshold_sdm, 'train/threshold_tdm': threshold_tdm
        })
        return loss_fm + loss_cl + loss_cr

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        self.shared_step(batch, self.valid_metric)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        self.shared_step(batch, self.test_metric)

    def shared_step(self, batch, metric):
        x, y = batch
        y_sd, y_td = F.softmax(self.sd_model(x), dim=1), F.softmax(self.td_model(x), dim=1)
        y_m = y_sd + y_td
        self.log_dict(metric((y_sd, y), (y_td, y), (y_m, y)), prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate_class(
            list(self.sd_model.parameters()) + list(self.td_model.parameters()) + list(self.loss_fn.parameters()),
        self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
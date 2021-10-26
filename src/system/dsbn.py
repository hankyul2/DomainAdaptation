import re
import os
import copy
from collections import OrderedDict

import torch
from torch import nn

from src.system.mstn import MSTN


class DSBN(nn.Module):
    def __init__(self, domain_name, bn):
        super().__init__()
        self.bns = nn.ModuleDict({name: copy.deepcopy(bn) for name in domain_name})
        self.current_domain = domain_name[0]

    def change_domain(self, domain_name):
        self.current_domain = domain_name
        return self

    def forward(self, x):
        return self.bns[self.current_domain](x)


def apply_dsbn(model, domain_name):
    dsbns = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
            name = re.sub('\\.(\\d)', lambda x: f'[{int(x.group(1))}]', name)
            module = DSBN(domain_name, module)
            exec(f'model.{name} = module')
            dsbns.append(module)
    model.dsbns = dsbns
    model.change_domain = lambda domain_name: [dsbn.change_domain(domain_name) for dsbn in dsbns]


def get_pseudo_label(student_y_hat, teacher_y_hat, ratio):
    return (student_y_hat * ratio + teacher_y_hat * (1 - ratio)).argmax(dim=1)


class DSBN_MSTN_Stage1(MSTN):
    def __init__(self, *args, **kwargs):
        super(DSBN_MSTN_Stage1, self).__init__(*args, **kwargs)
        apply_dsbn(self, ['src', 'tgt'])

    def get_feature(self, x, domain=None):
        self.change_domain(domain)
        return super(DSBN_MSTN_Stage1, self).get_feature(x, domain)

    def compute_loss_eval(self, x, y):
        self.change_domain('tgt')
        return super(DSBN_MSTN_Stage1, self).compute_loss_eval(x, y)


class DSBN_MSTN_Stage2(DSBN_MSTN_Stage1):
    def __init__(self, *args, teacher_model_path, **kwargs):
        super(DSBN_MSTN_Stage2, self).__init__(*args, **kwargs)
        self.teacher_model_path = teacher_model_path

    def on_fit_start(self) -> None:
        c = copy.deepcopy
        teacher_weight = torch.load(os.path.join(self.teacher_model_path, self.trainer.datamodule.dataset_name + '.ckpt'), map_location='cpu')
        self.change_domain('tgt')
        self.teacher_model = nn.Sequential(OrderedDict([('backbone', c(self.backbone)), ('bottleneck', c(self.bottleneck)), ('fc', c(self.fc))]))
        self.teacher_model.load_state_dict(teacher_weight, strict=True)
        self.teacher_model.requires_grad_(False)

    def training_step(self, batch, batch_idx, optimizer_idx=None, child_compute_already=None):
        (x_s, y_s), (x_t, y_t) = batch
        embed_s, y_hat_s = self.get_feature(x_s, 'src')
        embed_t, y_hat_t = self.get_feature(x_t, 'tgt')
        pseudo = get_pseudo_label(y_hat_t, self.teacher_model(x_t), self.get_alpha())
        loss = self.criterion(y_hat_s, y_s) + self.criterion(y_hat_t, pseudo)

        metric = self.train_metric(y_hat_s, y_s)
        self.log_dict({f'train/loss': loss})
        self.log_dict(metric)

        return loss

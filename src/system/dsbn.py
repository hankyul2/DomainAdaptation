from torch import nn
import copy
import re
import numpy as np

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


class DSBN_MSTN(MSTN):
    def __init__(self, *args, **kwargs):
        super(DSBN_MSTN, self).__init__(*args, **kwargs)
        apply_dsbn(self.backbone, ['src', 'tgt'])
        self.teacher_model = None

    def training_step(self, batch, batch_idx, optimizer_idx=None, child_compute_already=None):
        if self.current_epoch < 100:
            return super(DSBN_MSTN, self).training_step(batch, batch_idx, optimizer_idx, child_compute_already)
        else:
            if not self.teacher_model:
                self.generate_teacher_model()

            (x_s, y_s), (x_t, y_t) = batch
            embed_s, y_hat_s = self.get_feature(x_s, 'src')
            embed_t, y_hat_t = self.get_feature(x_t, 'tgt')
            pseudo = get_pseudo_label(y_hat_t, self.teacher_model(x_t), self.get_alpha())
            loss = self.criterion(y_hat_s, y_s) + self.criterion(y_hat_t, pseudo)

            metric = self.train_metric(y_hat_s, y_s)
            self.log_dict({f'train/loss': loss})
            self.log_dict(metric)

            return loss

    def generate_teacher_model(self):
        self.teacher_model = copy.deepcopy(nn.Sequential(self.backbone, self.bottleneck, self.fc))
        self.teacher_model.requires_grad_(False)
        self.teacher_model[0].change_domain('tgt')

    def get_feature(self, x, domain=None):
        self.backbone.change_domain(domain)
        return super(DSBN_MSTN, self).get_feature(x, domain)

    def compute_loss_eval(self, x, y):
        self.backbone.change_domain('tgt')
        return super(DSBN_MSTN, self).compute_loss_eval(x, y)

    def on_fit_end(self) -> None:
        del self.teacher_model

    def get_alpha(self):
        max_iter = self.num_step * self.max_epochs / 2
        if self.current_epoch < 100:
            return 2. / (1. + np.exp(-self.gamma * self.global_step / max_iter)) - 1
        else:
            return 2. / (1. + np.exp(-self.gamma * max(self.global_step - max_iter, 1) / max_iter)) - 1

import torch

from src.DomainModelWrapper import DomainModelWrapper
from src.cdan import get_model, conditional_entropy
from src.dataset import get_dataset, convert_to_dataloader
from src.log import get_log_name, Result
from src.resnet import get_resnet

from torch.optim import SGD, lr_scheduler as LR
import torch.nn.functional as F
from torch import nn

import numpy as np

from src.utils import AverageMeter


class ModelWrapper(DomainModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer, max_step, gamma=10):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.step = 0
        self.gamma = gamma
        self.max_step = max_step

    def forward(self, x_src, x_tgt, y_src):
        alpha = self.get_alpha()
        feat_src, cls_src, da_src = self.model(x_src, alpha)
        feat_tgt, cls_tgt, da_tgt = self.model(x_tgt, alpha)
        loss_cls, loss_da_src, loss_da_tgt = \
            self.criterion(cls_src, cls_tgt, da_src, da_tgt, y_src, alpha)
        loss = loss_cls + loss_da_src + loss_da_tgt

        self.cls_losses.update(loss_cls.item(), x_src.size(0))
        self.da_src_losses.update(loss_da_src.item(), x_src.size(0))
        self.da_tgt_losses.update(loss_da_tgt.item(), x_src.size(0))

        return loss, cls_src

    def init_progress(self, dl, epoch=None, mode='train'):
        super().init_progress(dl, epoch, mode)
        if mode == 'train':
            self.cls_losses = AverageMeter('CLS Loss', ':7.4f')
            self.da_src_losses = AverageMeter('DA(SRC) Loss', ':7.4f')
            self.da_tgt_losses = AverageMeter('DA(TGT) Loss', ':7.4f')
            self.progress.meters = [self.batch_time, self.data_time, self.losses, self.cls_losses,
                                    self.da_src_losses, self.da_tgt_losses, self.top1, self.top5]

    def get_alpha(self):
        self.step += 1
        return 2. / (1. + np.exp(-self.gamma * self.step / self.max_step)) - 1


class MyLoss(nn.Module):
    def __init__(self, use_entropy, alpha=1):
        super(MyLoss, self).__init__()
        if use_entropy:
            self.domain_criterion = lambda x, y, w, z: conditional_entropy(x, y, w, z)
        else:
            self.domain_criterion = lambda x, y, w, z: F.cross_entropy(x, y)
        self.alpha = alpha

    def forward(self, cls_src, cls_tgt, da_src, da_tgt, y_src, alpha):
        loss_cls = F.cross_entropy(cls_src, y_src)
        loss_da_src = self.domain_criterion(da_src, torch.ones(y_src.size(0)).long().to(y_src.device), cls_src, alpha) * self.alpha
        loss_da_tgt = self.domain_criterion(da_tgt, torch.zeros(y_src.size(0)).long().to(y_src.device), cls_tgt, alpha) * self.alpha
        return loss_cls, loss_da_src, loss_da_tgt


class MyOpt:
    def __init__(self, model, lr, nbatch, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': model.backbone.parameters(), 'lr': lr},
            {'params': model.bottleneck.parameters()},
            {'params': model.domain_classifier.parameters()},
            {'params': model.fc.parameters()}
        ], lr=lr * 10, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = LR.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.1)
        self.nbatch = nbatch
        self.step_ = 0

    def step(self):
        self.optimizer.step()
        self.step_ += 1
        if self.step_ % self.nbatch == 0:
            self.scheduler.step()
            self.step_ = 0

    def zero_grad(self):
        self.optimizer.zero_grad()


def run(args):
    # step 0. parse model name
    use_entropy = True if 'E' in args.model_name else False

    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, train=False)

    # step 2. prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_resnet()
    model = get_model(backbone, fc_dim=2048, embed_dim=1024, nclass=datasets[0].class_num, hidden_dim=1024).to(device)

    # step 3. training tool (criterion, optimizer)
    optimizer = MyOpt(model, lr=args.lr, nbatch=len(src_dl))
    criterion = MyLoss(use_entropy=use_entropy)

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model=model, device=device, optimizer=optimizer,
                         criterion=criterion, max_step=len(src_dl) * args.nepoch)
    best_dl = test_dl if args.use_ncrop_for_valid else None
    model.fit((src_dl, tgt_dl), valid_dl, test_dl=best_dl, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)
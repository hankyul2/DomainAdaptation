import copy

import torch

from src.DomainModelWrapper import DomainModelWrapper
from src.dann import get_model
from src.dataset import get_dataset, convert_to_dataloader
from src.fixbi import Fixbi
from src.log import get_log_name, Result
from src.resnet import get_resnet

from torch.optim import SGD, lr_scheduler as LR
import torch.nn.functional as F
from torch import nn

import numpy as np

from src.utils import AverageMeter


class ModelWrapper(DomainModelWrapper):
    def __init__(self, log_name, model_sdm, model_tdm, device, criterion, optimizer):
        super().__init__(log_name)
        self.model_sdm = model_sdm
        self.model_tdm = model_tdm
        self.model = model_tdm
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x_src, x_tgt, y_src, epoch):
        (loss_fm, loss_sp, loss_bim, loss_cr), cls_src = \
            self.criterion(x_src, x_tgt, y_src, self.model_sdm.predict, self.model_tdm.predict, epoch)
        loss = loss_fm + loss_sp + loss_bim + loss_cr

        self.fm_losses.update(loss_fm.item(), x_src.size(0))
        self.sp_losses.update(loss_sp.item(), x_src.size(0))
        self.bim_losses.update(loss_bim.item(), x_src.size(0))
        self.cr_losses.update(loss_cr.item(), x_src.size(0))

        return loss, cls_src

    def init_progress(self, dl, epoch=None, mode='train'):
        super().init_progress(dl, epoch, mode)
        if mode == 'train':
            self.fm_losses = AverageMeter('FM Loss', ':7.4f')
            self.sp_losses = AverageMeter('SP Loss', ':7.4f')
            self.bim_losses = AverageMeter('BIM Loss', ':7.4f')
            self.cr_losses = AverageMeter('CR Loss', ':7.4f')
            self.progress.meters = [self.batch_time, self.data_time, self.losses, self.fm_losses,
                                    self.sp_losses, self.bim_losses, self.cr_losses, self.top1, self.top5]


class MyOpt:
    def __init__(self, model_sdm, model_tdm, criterion, lr, nbatch, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': model_sdm.backbone.parameters()},
            {'params': model_sdm.bottleneck.parameters()},
            {'params': model_sdm.fc.parameters()},
            {'params': model_tdm.backbone.parameters()},
            {'params': model_tdm.bottleneck.parameters()},
            {'params': model_tdm.fc.parameters()},
            {'params': criterion.parameters()}
        ], lr=lr, momentum=momentum, weight_decay=weight_decay)
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
    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, train=False)

    # step 2. prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_resnet()
    model_sdm = get_model(backbone, fc_dim=2048, embed_dim=1024, nclass=datasets[0].class_num, hidden_dim=1024,
                      src=args.src, tgt=args.tgt).to(device)
    model_tdm = copy.deepcopy(model_sdm)

    # step 3. training tool (criterion, optimizer)
    criterion = Fixbi()
    optimizer = MyOpt(model_sdm, model_tdm, criterion, lr=args.lr, nbatch=len(src_dl))

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model_sdm=model_sdm, model_tdm=model_tdm, device=device,
                         optimizer=optimizer, criterion=criterion)
    best_dl = test_dl if args.use_ncrop_for_valid else None
    model.fit((src_dl, tgt_dl), valid_dl, test_dl=best_dl, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)
import copy

import torch

from src.domain_model_wrapper import DomainModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.loss.fixbi import Fixbi
from src.log import get_log_name, Result
from src.model.models import get_model

from torch.optim import SGD

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

    def forward(self, x_src, x_tgt, y_src, epoch=None):
        (loss_fm, loss_sp, loss_bim, loss_cr), cls_src, (sdm_threshold, tdm_threshold) = \
            self.criterion(x_src, x_tgt, y_src, self.model_sdm.predict, self.model_tdm.predict, epoch)
        loss = loss_fm + loss_sp + loss_bim + loss_cr

        self.fm_losses.update(loss_fm.item(), x_src.size(0))
        self.sp_losses.update(loss_sp.item(), x_src.size(0))
        self.bim_losses.update(loss_bim.item(), x_src.size(0))
        self.cr_losses.update(loss_cr.item(), x_src.size(0))

        self.sdm_threshold.update(sdm_threshold.item(), x_src.size(0))
        self.tdm_threshold.update(tdm_threshold.item(), x_src.size(0))

        return loss, cls_src

    def init_progress(self, dl, epoch=None, mode='train'):
        super().init_progress(dl, epoch, mode)
        if mode == 'train':
            if epoch > 0:
                self.writer.add_scalar('Threshold/SDM', self.sdm_threshold.avg, epoch)
                self.writer.add_scalar('Threshold/TDM', self.tdm_threshold.avg, epoch)

            self.sdm_threshold = AverageMeter('SDM Thrshold', ':.4f')
            self.tdm_threshold = AverageMeter('TDM Thrshold', ':.4f')

            self.fm_losses = AverageMeter('FM Loss', ':7.4f')
            self.sp_losses = AverageMeter('SP Loss', ':7.4f')
            self.bim_losses = AverageMeter('BIM Loss', ':7.4f')
            self.cr_losses = AverageMeter('CR Loss', ':7.4f')
            self.progress.meters = [self.batch_time, self.data_time, self.losses, self.fm_losses,
                                    self.sp_losses, self.bim_losses, self.cr_losses, self.top1, self.top5]


class MyOpt:
    def __init__(self, model_sdm, model_tdm, criterion, lr, nepoch, nbatch, weight_decay=0.005, momentum=0.9):
        self.optimizer_sdm = SGD([
            {'params': model_sdm.backbone.parameters(), 'lr0': lr * 0.1},
            {'params': model_sdm.bottleneck.parameters(), 'lr0': lr},
            {'params': model_sdm.fc.parameters(), 'lr0': lr},
            {'params': criterion.T_sdm, 'lr0': lr}
        ], lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_tdm = SGD([
            {'params': model_tdm.backbone.parameters(), 'lr0': lr * 0.1},
            {'params': model_tdm.bottleneck.parameters(), 'lr0': lr},
            {'params': model_tdm.fc.parameters(), 'lr0': lr},
            {'params': criterion.T_tdm, 'lr0': lr}
        ], lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.total_step_ = nepoch * nbatch
        self.step_ = 0

    def step(self):
        self.step_ += 1
        decay = self.get_decay()

        for param in self.optimizer_sdm.param_groups + self.optimizer_tdm.param_groups:
            param['lr'] = param['lr0'] * decay

        self.optimizer_sdm.step()
        self.optimizer_tdm.step()

    def get_decay(self, step=None):
        step = step if step else self.step_
        return (1.0 + 10.0 * step/self.total_step_) ** (-0.75)

    def zero_grad(self):
        self.optimizer_sdm.zero_grad()
        self.optimizer_tdm.zero_grad()


def run(args):
    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, shuffle=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, shuffle=False)

    # step 2. prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_sdm = get_model('DANN', nclass=datasets[0].class_num, src=args.src, tgt=args.tgt).to(device)
    model_tdm = copy.deepcopy(model_sdm)

    # step 3. training tool (criterion, optimizer)
    criterion = Fixbi()
    optimizer = MyOpt(model_sdm, model_tdm, criterion, lr=args.lr, nepoch=args.nepoch, nbatch=len(src_dl))

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model_sdm=model_sdm, model_tdm=model_tdm, device=device,
                         optimizer=optimizer, criterion=criterion)
    best_dl = test_dl if args.use_ncrop_for_valid else None
    model.fit((src_dl, tgt_dl), valid_dl, test_dl=best_dl, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)
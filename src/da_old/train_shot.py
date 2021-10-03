import torch
from torch.optim import SGD, lr_scheduler as LR
from torch import nn
import torch.nn.functional as F

from src.base_model_wrapper import BaseModelWrapper
from src.model.basic import get_model
from src.dataset import get_dataset, convert_to_dataloader
from src.common_module import entropy, divergence, LabelSmoothing
from src.log import Result
from src.model.resnet import get_resnet

from src.utils import AverageMeter


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, start_time, model, device, criterion, optimizer):
        super().__init__(log_name, start_time)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x_tgt, y_tgt, epoch=None):
        cls_tgt = self.model(x_tgt)
        loss_pseudo, loss_entropy, loss_div = \
            self.criterion(cls_tgt)
        loss = loss_pseudo + loss_entropy + loss_div

        self.pseudo_losses.update(loss_pseudo.item(), x_tgt.size(0))
        self.entropy_losses.update(loss_entropy.item(), x_tgt.size(0))
        self.div_losses.update(loss_div.item(), x_tgt.size(0))

        return loss, cls_tgt

    def init_progress(self, dl, epoch=None, mode='train'):
        super().init_progress(dl, epoch, mode)
        if mode == 'train':
            self.pseudo_losses = AverageMeter('CLS Loss', ':7.4f')
            self.entropy_losses = AverageMeter('DA(SRC) Loss', ':7.4f')
            self.div_losses = AverageMeter('DA(TGT) Loss', ':7.4f')
            self.progress.meters = [self.batch_time, self.data_time, self.losses, self.pseudo_losses,
                                    self.entropy_losses, self.div_losses, self.top1, self.top5]


class MyLoss(nn.Module):
    def __init__(self, alpha=1):
        super(MyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, cls_tgt):
        loss_entropy = entropy(F.softmax(cls_tgt))
        loss_div = divergence(F.softmax(cls_tgt))
        pseudo_label = None
        loss_cls = LabelSmoothing(cls_tgt, pseudo_label)
        return loss_cls, loss_entropy, loss_div


class MyOpt:
    def __init__(self, model, lr, nbatch, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': model.bottleneck.parameters()},
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
    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, shuffle=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, shuffle=False)

    # step 2. prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_resnet()
    model = get_model(backbone, fc_dim=2048, embed_dim=1024, nclass=datasets[0].class_num, hidden_dim=1024,
                      src=args.src, tgt=args.tgt).to(device)

    # step 3. training tool (criterion, optimizer)
    optimizer = MyOpt(model, lr=args.lr, nbatch=len(src_dl))
    criterion = MyLoss()

    # step 4. train
    model = ModelWrapper(log_name=args.log_name, start_time=args.start_time, model=model, device=device,
                         optimizer=optimizer, criterion=criterion)
    best_dl = test_dl if args.use_ncrop_for_valid else None
    model.fit(tgt_dl, valid_dl, test_dl=best_dl, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)

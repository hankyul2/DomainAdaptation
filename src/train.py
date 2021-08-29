import torch
from easydict import EasyDict as edict
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.basicModel import get_model
from src.resnet import get_resnet
from src.ModelWrapper import BaseModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.log import get_log_name, Result


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


class MyOpt:
    def __init__(self, model, lr, nbatch, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': model.backbone.parameters(), 'lr': lr},
            {'params': model.bottleneck.parameters()},
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
    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, train=False)

    # step 2. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_resnet()
    model = get_model(backbone, fc_dim=2048, embed_dim=1024, nclass=datasets[0].class_num).to(device)

    # step 3. prepare training tool
    criterion = nn.CrossEntropyLoss()
    optimizer = MyOpt(model, lr=args.lr, nbatch=len(src_dl))

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model=model, device=device, optimizer=optimizer, criterion=criterion)
    model.fit(src_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)


if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
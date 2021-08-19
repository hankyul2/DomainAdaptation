import torch

from src.ModelWrapper import BaseModelWrapper
from src.bsp import BSP
from src.cdan import get_model
from src.dataset import get_dataset, convert_to_dataloader
from src.resnet import get_resnet

from torch.optim import SGD, lr_scheduler as LR
import torch.nn.functional as F
from torch import nn

import numpy as np


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer, max_step, gamma=10):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.step = 0
        self.gamma = gamma
        self.max_step = max_step

    def train(self, train_dl):
        self.model.train()
        src_dl, tgt_dl = train_dl
        debug_step, total_step, total_loss, total_acc = 10, len(src_dl), 0, 0

        for step, ((x_src, y_src), (x_tgt, y_tgt)) in enumerate(zip(src_dl, tgt_dl)):
            ##########################################################
            alpha = self.get_alpha()
            x_src, x_tgt, y_src = x_src.to(self.device), x_tgt.to(self.device), y_src.long().to(self.device)
            feat_src, cls_src, da_src = self.model(x_src, alpha)
            feat_tgt, cls_tgt, da_tgt = self.model(x_tgt, alpha)
            loss_bsp, loss_cls, loss_da_src, loss_da_tgt = self.criterion(cls_src, da_src, da_tgt, feat_src, feat_tgt,
                                                                          y_src)
            loss = loss_bsp + loss_cls + loss_da_src + loss_da_tgt
            ##########################################################
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            _, y_label = cls_src.max(dim=1)
            acc = (y_label == y_src).sum().item() / len(y_src)
            total_loss += loss.clone().detach().item()
            total_acc += acc

            if step != 0 and step % debug_step == 0:
                self.log(
                    '[train] STEP: {:03d}/{:03d}  |  total loss {:07.4f}  |  cls loss {:07.4f}  '
                    '|  da src loss {:07.4f}  |  da tgt loss {:07.4f}  |  bsp loss {:07.4f}  |  acc {:07.4f}%'.
                        format(step + 1, total_step, loss.clone().detach().item(), loss_cls.clone().detach().item(),
                               loss_da_src.clone().detach().item(), loss_da_tgt.clone().detach().item(),
                               loss_bsp.clone().detach().item(), acc * 100
                               ))

        return total_loss / total_step, total_acc / total_step

    def get_alpha(self):
        self.step += 1
        return 2. / (1. + np.exp(-self.gamma * self.step / self.max_step)) - 1


class MyLoss(nn.Module):
    def __init__(self, alpha=1, beta=1e-4):
        super(MyLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, cls_src, da_src, da_tgt, feat_src, feat_tgt, y_src):
        loss_bsp = BSP(feat_src, feat_tgt) * self.beta
        loss_cls = F.cross_entropy(cls_src, y_src)
        loss_da_src = F.cross_entropy(da_src, torch.ones(y_src.size(0)).long().to(y_src.device)) * self.alpha
        loss_da_tgt = F.cross_entropy(da_tgt, torch.zeros(y_src.size(0)).long().to(y_src.device)) * self.alpha
        return loss_bsp, loss_cls, loss_da_src, loss_da_tgt


class MyOpt:
    def __init__(self, model, lr, nbatch, nepoch=50, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD([
            {'params': model.backbone.parameters(), 'lr': lr},
            {'params': model.bottleneck.parameters()},
            {'params': model.domain_classifier.parameters()},
            {'params': model.fc.parameters()}
        ], lr=lr * 10, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = LR.StepLR(self.optimizer, nbatch * (nepoch / 3), 0.1)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def run(log_name='cdan-bsp', src='amazon', tgt='webcam', batch_size=32, num_workers=4, lr=0.0003, nepoch=50, ncrop=10):
    # step 1. prepare dataset
    datasets = get_dataset(src, tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], batch_size, num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], batch_size, num_workers, train=False)

    # step 2. prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_resnet(device)
    model = get_model(backbone, fc_dim=2048, embed_dim=256, nclass=31, hidden_dim=1024).to(device)

    # step 3. training tool (criterion, optimizer)
    optimizer = MyOpt(model, lr=lr, nbatch=len(src_dl), nepoch=nepoch)
    criterion = MyLoss()

    # step 4. train
    model = ModelWrapper(log_name, model=model, device=device, optimizer=optimizer, criterion=criterion,
                         max_step=len(src) * nepoch)
    model.fit((src_dl, tgt_dl), valid_dl, nepoch)

    # step 5. evaluation
    model.load_best_weight()
    model.evaluate(test_dl, ncrop=ncrop)


if __name__ == '__main__':
    run(src='webcam', tgt='amazon')

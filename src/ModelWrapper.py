import time
import datetime
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BaseModelWrapper:
    def __init__(self, log_name):
        self.best_acc = 0
        self.model = None
        self.device = None
        self.logger = logging.getLogger()
        self.log_name = '{}-{}'.format(datetime.datetime.now(), log_name)
        self.log_best_weight_path = 'log_best_weight/{}'.format(self.log_name)
        self.writer = SummaryWriter(log_dir='log_tensor_board/{}'.format(self.log_name), filename_suffix=log_name)

        self.setup_directory()
        self.setup_file_logger('log_text/{}-{}'.format(datetime.datetime.now(), log_name))


    def setup_directory(self):
        Path('log_text').mkdir(exist_ok=True, parents=True)
        Path('log_tensor_board').mkdir(exist_ok=True, parents=True)
        Path('log_best_weight').mkdir(exist_ok=True, parents=True)

    def setup_file_logger(self, log_file):
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        print('{} {}'.format(datetime.datetime.now(), message))
        self.logger.info(message)

    def log_tensorboard(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/test', valid_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/test', valid_acc, epoch)

    def save_best_weight(self, model, top1_acc):
        if top1_acc > self.best_acc:
            self.best_acc = top1_acc
            self.log('saving best model({:07.4f}%) weight to {}'.format(top1_acc*100, self.log_best_weight_path))
            torch.save({'weight':model.state_dict(), 'top1_acc':top1_acc}, self.log_best_weight_path)

    def load_best_weight(self, path=None):
        path = path if path else self.log_best_weight_path
        if self.model is not None and self.device is not None:
            self.model.load_state_dict(torch.load(f=path, map_location=self.device)["weight"])

    @torch.no_grad()
    def valid(self, dl):
        self.model.eval()
        debug_step, total_step, total_loss, total_acc = 10, len(dl), 0, 0
        for step, (x, y) in enumerate(dl):
            x, y = x.float().to(self.device), y.long().to(self.device)
            y_hat = self.model.predict(x)
            loss = F.cross_entropy(y_hat, y)

            _, y_label = y_hat.max(dim=1)
            acc = (y_label == y).sum().item() / len(y)

            total_loss += loss.clone().detach().item()
            total_acc += acc

            if step % debug_step == 0:
                self.log('[valid] STEP: {:03d}/{:03d}  |  loss {:07.4f} acc {:07.4f}%'.
                         format(step + 1, total_step, loss.clone().detach().item(), acc * 100))

        return total_loss / total_step, total_acc / total_step

    def fit(self, train_dl, valid_dl, nepoch=50):
        best_acc = 0
        best_acc_arg = 0

        for epoch in range(nepoch):
            train_loss, train_acc = self.train(train_dl)
            valid_loss, valid_acc = self.valid(valid_dl)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_acc_arg = epoch + 1
                self.save_best_weight(self.model, best_acc)

            print('=' * 180)
            self.log(
                '[EPOCH]: {:03d}/{:03d}  |  train loss {:07.4f} acc {:07.4f}%  |  valid loss {:07.4f} acc {:07.4f}% ('
                'best accuracy : {:07.4f} @ {:03d})'.format(
                    epoch + 1, nepoch, train_loss, train_acc * 100, valid_loss, valid_acc * 100, best_acc * 100,
                    best_acc_arg
                ))
            print('=' * 180)
            self.log_tensorboard(epoch, train_loss, train_acc * 100, valid_loss, valid_acc * 100)

    @torch.no_grad()
    def evaluate(self, test_dl, ncrop=10):
        self.model.eval()
        total_loss = total_acc = total_y_hat = 0
        for iter in range(ncrop):
            y_hat = []
            total_y = []
            for x, y in test_dl:
                x, y = x.float().to(self.device), y.long().to(self.device)
                y_hat.append(self.model.predict(x))
                total_y.append(y)
            total_y_hat += torch.cat(y_hat, dim=0)
            total_y = torch.cat(total_y, dim=0)

        total_loss = F.cross_entropy(total_y_hat/ncrop, total_y).clone().detach().item()
        _, y_label = total_y_hat.max(dim=1)
        total_acc = (y_label == total_y).sum().item() / len(total_y)

        print('=' * 180)
        self.log('[EVALUATE] {}-crop loss {:07.4f}  |  {}-crop acc {:07.4f}%'.format(ncrop, total_loss, ncrop, total_acc * 100))
        print('=' * 180)



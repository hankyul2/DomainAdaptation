import time

from src.base_model_wrapper import BaseModelWrapper
from src.utils import accuracy


class DomainModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, start_time):
        super(DomainModelWrapper, self).__init__(log_name, start_time)

    def forward(self, x_src, x_tgt, y_src, epoch=None):
        raise NotImplementedError

    def train(self, train_dl, epoch):
        src_dl, tgt_dl = train_dl
        debug_step = len(src_dl) // 10
        self.init_progress(src_dl, epoch=epoch, mode='train')
        self.model.train()

        end = time.time()
        for step, ((x_src, y_src), (x_tgt, y_tgt)) in enumerate(zip(src_dl, tgt_dl)):
            self.data_time.update(time.time() - end)

            x_src, x_tgt, y_src = x_src.to(self.device), x_tgt.to(self.device), y_src.long().to(self.device)
            loss, std_y_hat = self.forward(x_src, x_tgt, y_src, epoch)

            acc1, acc5 = accuracy(std_y_hat, y_src, topk=(1, 5))
            self.losses.update(loss.item(), x_src.size(0))
            self.top1.update(acc1[0], x_src.size(0))
            self.top5.update(acc5[0], x_src.size(0))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.batch_time.update(time.time() - end)
            end = time.time()

            if step != 0 and step % debug_step == 0:
                self.log(self.progress.display(step))

        return self.losses.avg, self.top1.avg

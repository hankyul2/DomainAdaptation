from torch.optim import SGD
import torch.optim.lr_scheduler as LR


def apply_target_lr(model, lr, target_factor=0.1, target_name='backbone'):
    """apply target factor lr only to target name module"""
    params = []
    for name, module in model.named_children():
        target_lr = (lr * target_factor) if target_name in name else lr
        params.append({'params':module.parameters(), 'lr0': target_lr, 'lr': target_lr})
    return params


class MultiOpt:
    def __init__(self, model, lr, nbatch, nepoch, weight_decay=1e-3, momentum=0.9):
        self.optimizer = SGD(apply_target_lr(model, lr), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        self.max_iter = nbatch * nepoch
        self.step_ = 0

    def step(self):
        self.optimizer.step()
        self.step_ += 1
        decay = self.get_decay()
        for param in self.optimizer.param_groups:
            param['lr'] = param['lr0'] * decay

    def get_decay(self, step=None):
        step = step if step else self.step_
        return (1 + 10 * step / self.max_iter) ** (-0.75)

    def zero_grad(self):
        self.optimizer.zero_grad()


class StepOpt:
    def __init__(self, model, lr, nbatch, weight_decay=0.0005, momentum=0.95):
        self.optimizer = SGD(apply_target_lr(model, lr), lr=lr, momentum=momentum, weight_decay=weight_decay)
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

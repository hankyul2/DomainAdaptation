import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ])
        self.fc = nn.Linear(64, 16)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)


class Fixbi(nn.Module):
    def __init__(self, warmup_epoch=25, lambda_src=0.7, lambda_tgt=0.3, lambda_mid=0.5):
        super(Fixbi, self).__init__()
        self.warmup_epoch = warmup_epoch
        self.lambda_src = lambda_src
        self.lambda_tgt = lambda_tgt
        self.lambda_mid = lambda_mid
        self.T_sdm = nn.Parameter(torch.tensor(5.0))
        self.T_tdm = nn.Parameter(torch.tensor(5.0))

    def mixup(self, src, tgt, ratio):
        return src * ratio + tgt * (1 - ratio)

    def get_pseudo_label(self, model, x):
        y_hat = model(x)
        y_prob, y_pred = F.softmax(y_hat, dim=1).max(dim=1)
        mean, std = y_prob.mean(), y_prob.std()
        threshold = mean - 2 * std
        return y_hat, y_prob, y_pred, threshold

    def mixup_criterion(self, y_hat, y_src, y_pseudo, ratio):
        return F.cross_entropy(y_hat, y_src.detach()) * ratio + \
               F.cross_entropy(y_hat, y_pseudo.detach()) * (1 - ratio)

    def self_penalization(self, x, target, T):
        return F.nll_loss(torch.log(1 - F.softmax(x / T, dim=1)), target.detach())

    def bidirectional_matching(self, sdm_hat, sdm_pseudo, tdm_hat, tdm_pseudo):
        return F.cross_entropy(sdm_hat, tdm_pseudo.detach()) + F.cross_entropy(tdm_hat, sdm_pseudo.detach())

    def get_minimum_len(self, sdm_low_prediction, tdm_low_prediction):
        if sdm_low_prediction.dim() > 0 and tdm_low_prediction.dim() > 0:
            if sdm_low_prediction.numel() > 0 and tdm_low_prediction.numel() > 0:
                return min(sdm_low_prediction.size(0), tdm_low_prediction.size(0))
            else:
                return 0
        else:
            return 0

    def apply_mask(self, sdm_tgt, tdm_tgt, threshold_fn, loss_fn):
        sdm_tgt_hat, sdm_tgt_prob, sdm_tgt_pseudo, sdm_tgt_threshold = sdm_tgt
        tdm_tgt_hat, tdm_tgt_prob, tdm_tgt_pseudo, tdm_tgt_threshold = tdm_tgt

        sdm_mask = torch.nonzero(threshold_fn(sdm_tgt_prob, sdm_tgt_threshold), as_tuple=False).squeeze()
        tdm_mask = torch.nonzero(threshold_fn(tdm_tgt_prob, tdm_tgt_threshold), as_tuple=False).squeeze()

        mask_len = self.get_minimum_len(sdm_mask, tdm_mask)

        if mask_len > 0:
            sdm_mask = sdm_mask[:mask_len]
            tdm_mask = tdm_mask[:mask_len]

            sdm_hat = sdm_tgt_hat[sdm_mask]
            tdm_hat = tdm_tgt_hat[tdm_mask]

            sdm_pseudo = sdm_tgt_pseudo[sdm_mask]
            tdm_pseudo = tdm_tgt_pseudo[tdm_mask]

            return loss_fn(sdm_hat, sdm_pseudo, tdm_hat, tdm_pseudo)
        else:
            return torch.tensor(0.0)

    def forward(self, x_src, x_tgt, y_src, sdm, tdm, epoch):
        loss_fm = loss_sp = loss_bim = loss_cr = torch.tensor(0.0)

        # step 0. compute pseudo label
        _, _, sdm_tgt_pseudo, _ = sdm_tgt = self.get_pseudo_label(sdm, x_tgt)
        _, _, tdm_tgt_pseudo, _ = tdm_tgt = self.get_pseudo_label(tdm, x_tgt)

        # step 1 : fixed ratio mixup loss
        x_sd = self.mixup(x_src, x_tgt, self.lambda_src)
        x_td = self.mixup(x_src, x_tgt, self.lambda_tgt)
        y_sd = sdm(x_sd)
        y_td = tdm(x_td)
        loss_fm = self.mixup_criterion(y_sd, y_src, sdm_tgt_pseudo, self.lambda_src) + \
                  self.mixup_criterion(y_td, y_src, tdm_tgt_pseudo, self.lambda_tgt)

        if epoch < self.warmup_epoch:
            # step 2 : self_penalization (epoch < warmup_epoch)
            loss_sp = self.apply_mask(sdm_tgt, tdm_tgt, torch.lt,
                                      lambda sdm_x, sdm_y, tdm_x, tdm_y:
                                      self.self_penalization(sdm_x, sdm_y, self.T_sdm) +
                                       self.self_penalization(tdm_x, tdm_y, self.T_tdm))
        else:
            # step 3 : bidirectional matching
            loss_bim = self.apply_mask(sdm_tgt, tdm_tgt, torch.gt,
                                       lambda sdm_x, sdm_y, tdm_x, tdm_y:
                                       self.bidirectional_matching(sdm_x, sdm_y, tdm_x, tdm_y))

            # step 4 : consistency regularization
            x_mid = self.mixup(x_src, x_tgt, self.lambda_mid)
            loss_cr = F.mse_loss(sdm(x_mid), tdm(x_mid))

        return (loss_fm, loss_sp, loss_bim, loss_cr), y_sd



if __name__ == '__main__':
    fixbi = Fixbi()

    x_src = torch.rand((16, 3, 224, 224))
    x_tgt = torch.rand((16, 3, 224, 224))
    y_src = torch.arange(16)

    sdm = Model()
    tdm = Model()

    epoch = 11

    loss_fm, loss_sp, loss_bim, loss_cr = fixbi(x_src, x_tgt, y_src, sdm, tdm, epoch)

    print(loss_fm)
    print(loss_sp)
    print(loss_bim)
    print(loss_cr)
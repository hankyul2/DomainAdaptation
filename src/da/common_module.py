import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from src.loss.im import entropy


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        x = self.layer(x)
        return x


def conditional_entropy(pred_dom, y_dom, pred_cls, alpha):
    # Todo: improve E performance
    pre_cls_softmax = F.softmax(pred_cls, dim=-1)
    e = GRL.apply(entropy(pre_cls_softmax), alpha)
    w = 1 + torch.exp(-e)
    loss = F.cross_entropy(pred_dom, y_dom, reduction='none')
    conditional_loss = ((w/w.sum(dim=0).detach().item()) * loss).sum(dim=0)
    return conditional_loss



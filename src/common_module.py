import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


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


def entropy(prediction_softmax, eps=1e-5):
    return (prediction_softmax * -torch.log(prediction_softmax+eps)).sum(dim=1)


def divergence(prediction_softmax, eps=1e-5):
    p = prediction_softmax.mean(dim=0)
    return (p * -torch.log(p + eps)).sum()


class LabelSmoothing(nn.Module):
    def __init__(self, alpha=0.1):
        super(LabelSmoothing, self).__init__()
        self.alpha = alpha
        self.certainty = 1.0 - alpha
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, y):
        b, c = x.shape
        label = torch.full((b, c), self.alpha/(c-1)).to(y.device)
        label = label.scatter_(1, y.unsqueeze(1), self.certainty)
        return self.criterion(F.log_softmax(x, dim=-1), label)
import torch
from torch import nn
import torch.nn.functional as F

from src.common_module import DomainClassifier, GRL, entropy
from src.system.dann import DANN


class CDAN(DANN):
    def __init__(self, *args, **kwargs):
        super(CDAN, self).__init__(*args, **kwargs)
        self.cdan_dim = kwargs['embed_dim'] * kwargs['num_classes']
        self.dc = DomainClassifier(self.cdan_dim, kwargs['hidden_dim'])

    def compute_dc_loss(self, embed_s, embed_t, y_hat_s, y_hat_t):
        c_embed_s = self.conditional_embed(embed_s, y_hat_s, embed_s.size(0))
        c_embed_t = self.conditional_embed(embed_t, y_hat_t, embed_t.size(0))
        return super(CDAN, self).compute_dc_loss(c_embed_s, c_embed_t, y_hat_s, y_hat_t)

    def conditional_embed(self, embed, y_hat, batch_size):
        return (F.softmax(y_hat, dim=1).detach().unsqueeze(2) @ embed.unsqueeze(1)).view(batch_size, self.cdan_dim)


class CDAN_E(CDAN):
    def __init__(self, *args, **kwargs):
        super(CDAN_E, self).__init__(*args, **kwargs)
        self.criterion_dc = nn.CrossEntropyLoss(reduction='none')

    def compute_dc_loss(self, embed_s, embed_t, y_hat_s, y_hat_t):
        dc_loss = super(CDAN_E, self).compute_dc_loss(embed_s, embed_t, y_hat_s, y_hat_t)
        return self.conditional_dc_loss(dc_loss, torch.cat([y_hat_s, y_hat_t]))

    def conditional_dc_loss(self, dc_loss, y_hat_cls):
        e = GRL.apply(entropy(F.softmax(y_hat_cls, dim=-1)), self.get_alpha())
        w = 1 + torch.exp(-e)
        return ((w / w.sum(dim=0).detach().item()) * dc_loss).sum(dim=0)
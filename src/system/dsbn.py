from torch import nn
import copy
import re

from src.system.mstn import MSTN


class DSBN(nn.Module):
    def __init__(self, domain_name, bn):
        super().__init__()
        self.bns = nn.ModuleDict({name: copy.deepcopy(bn) for name in domain_name})
        self.current_domain = domain_name[0]

    def change_domain(self, domain_name):
        self.current_domain = domain_name
        return self

    def forward(self, x):
        return self.bns[self.current_domain](x)


def apply_dsbn(model, domain_name):
    dsbns = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
            name = re.sub('\\.(\\d)', lambda x: f'[{int(x.group(1))}]', name)
            module = DSBN(domain_name, module)
            exec(f'model.{name} = module')
            dsbns.append(module)
    model.dsbns = dsbns
    model.change_domain = lambda domain_name: [dsbn.change_domain(domain_name) for dsbn in dsbns]
    

class DSBN_MSTN(MSTN):
    def __init__(self, *args, **kwargs):
        super(DSBN_MSTN, self).__init__(*args, **kwargs)
        apply_dsbn(self.backbone, ['src', 'tgt'])
        
    def get_feature(self, x, domain=None):
        self.backbone.change_domain(domain)
        return super(DSBN_MSTN, self).get_feature(x, domain)
    
    def compute_loss_eval(self, x, y):
        self.backbone.change_domain('tgt')
        return super(DSBN_MSTN, self).compute_loss_eval(x, y)
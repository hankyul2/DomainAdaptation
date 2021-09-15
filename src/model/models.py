import torch

from src.log import Result
from src.model.cdan import get_cdan
from src.model.dann import get_dann
from src.model.resnet import get_resnet
from src.model.basic import get_basic_model


def get_model(model_name, nclass, device, src=None, tgt=None, **kwargs):
    backbone = get_resnet(pretrained=True)

    if model_name == 'BASE':
        model = get_basic_model(backbone, nclass=nclass, **kwargs)
    elif model_name == 'DANN':
        model = get_dann(backbone, nclass=nclass, **kwargs)
    elif model_name == 'CDAN':
        model = get_cdan(backbone, nclass=nclass, **kwargs)


    if src and tgt:
        model.load_state_dict(torch.load(
            Result().get_best_pretrained_model_path(model_name, src, tgt), map_location='cpu')['weight'])


    return model.to(device)
from pathlib import Path

import torchvision.models as models
from torch import nn
from torch.hub import load_state_dict_from_url


def load_from_zoo(pretrained_model_save_path="pretrained_model_weight"):
    Path(pretrained_model_save_path).mkdir(exist_ok=True)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', progress=True,
                                          model_dir=pretrained_model_save_path, map_location='cpu')
    return state_dict


def get_resnet(pretrained=True):
    net = models.resnet50(pretrained=False)
    net.fc = nn.Flatten()

    if pretrained:
        state_dict = load_from_zoo()
        net.load_state_dict(state_dict, strict=False)
    return net
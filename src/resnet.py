from pathlib import Path

import torchvision.models as models
from torch import nn
from torch.hub import load_state_dict_from_url


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()


def get_resnet(device, pretrained_model_save_path="pretrained_model_weight"):
    Path(pretrained_model_save_path).mkdir(exist_ok=True)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', progress=True, model_dir=pretrained_model_save_path)
    net = models.resnet50(pretrained=False)
    net.load_state_dict(state_dict)
    net.fc = Identity()
    net = net.to(device)
    net.eval()
    return net
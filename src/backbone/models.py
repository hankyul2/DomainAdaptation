from src.backbone.hybrid import get_hybrid
from src.backbone.mobile_net_v2 import get_mobilenet_v2
from src.backbone.resnet import get_resnet
from src.backbone.resnet32 import get_resnet32
from src.backbone.senet import get_seresnet
from src.backbone.vit import get_vit


def get_model(model_name: str, **kwargs):
    if model_name.startswith('resnet32'):
        model = get_resnet32(model_name, **kwargs)
    elif model_name.startswith('resnet'):
        model = get_resnet(model_name, **kwargs)
    elif model_name.startswith('vit'):
        model = get_vit(model_name, **kwargs)
    elif model_name.startswith('r50'):
        model = get_hybrid(model_name, **kwargs)
    elif model_name.startswith('mobilenet_v2'):
        model = get_mobilenet_v2('mobilenet_v2', **kwargs)
    elif model_name.startswith('seresnet'):
        model = get_seresnet(model_name, **kwargs)

    return model
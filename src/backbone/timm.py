import timm
import torch


def get_timm(model_name: str, pretrained=False, **kwargs):
    model = timm.create_model('vit_base_r50_s16_224_in21k', pretrained=pretrained)
    model.head = torch.nn.Identity()
    model.out_channels = 768

    if 'pretrained_path' in kwargs:
        model.load_state_dict(torch.load(kwargs['pretrained_path']), strict=False)

    return model
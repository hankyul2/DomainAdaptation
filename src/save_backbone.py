from collections import OrderedDict

import torch


ckpt_path = 'log/r50_vit_base_patch16_224_webcam/DA-938/epoch=11_acc=1.0000.ckpt'
state_dict_path = 'pretrained/source_only_vit/webcam.ckpt'
state_dict = OrderedDict()
for k, v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    print(k)
    if 'backbone' in k:
        state_dict[k] = v
    if 'fc' in k:
        state_dict[k] = v
    if 'bottleneck' in k:
        state_dict[k] = v

with open(state_dict_path, 'wb') as f:
    torch.save(state_dict,f)
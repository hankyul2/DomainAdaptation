from collections import OrderedDict

import torch


ckpt_path = 'log/resnet50_webcam_dslr/DA-855/epoch=14_acc=0.9980.ckpt'
state_dict_path = 'pretrained/dann/webcam_dslr.ckpt'
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
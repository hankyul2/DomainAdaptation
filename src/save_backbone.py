from collections import OrderedDict

import torch


ckpt_path = 'log/resnet50_webcam_amazon/DA-87/epoch=54_step=0.00_loss=0.072.ckpt'
state_dict_path = 'pretrained/da/webcam_amazon_bsp_cdan_e.pth'
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
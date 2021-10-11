import os

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import instantiate_class

from src.common_module import entropy, divergence
from src.system.source_only import DABase


class SHOT(DABase):
    def __init__(self, *args, source_only_path: str = None, **kwargs):
        super(SHOT, self).__init__(*args, **kwargs)
        self.source_only_path = source_only_path

    def on_fit_start(self) -> None:
        weight_path = os.path.join(self.source_only_path, self.trainer.datamodule.src+'.ckpt')
        self.load_state_dict(torch.load(weight_path)['state_dict'])

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()
        self.bottleneck.eval()
        self.fc.eval()

        with torch.no_grad():
            embed = []
            p = []
            train_tgt = self.trainer.datamodule.train_dataloader()[1].dataset
            test_tgt = self.trainer.datamodule.test_dataloader()
            for x, _ in test_tgt:
                embed.append(self.bottleneck(self.backbone(x.to(self.device))))
                p.append(F.softmax(self.fc(embed[-1]), dim=1))
            embed, p = torch.cat(embed, dim=0), torch.cat(p, dim=0)

            pseudo_label = self.cluster(embed, p.t())
            weight = torch.eye(self.num_classes) @ torch.eye(self.num_classes)[pseudo_label].t()
            pseudo_label = self.cluster(embed, weight.to(self.device))

            train_tgt.samples = [(train_tgt.samples[i][0], pseudo_label[i].item()) for i in range(len(train_tgt))]

        self.backbone.train()
        self.bottleneck.train()
        self.fc.train()

    def cluster(self, embed, weight):
        centroid = weight @ embed / weight.sum(dim=1, keepdim=True)
        return (F.normalize(embed) @ F.normalize(centroid).t()).max(dim=1)[1]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        src, tgt = batch
        return self.shared_step(tgt, self.train_metric, 'train', add_dataloader_idx=False)

    def compute_loss(self, x, y):
        cls_loss, y_hat = self.compute_loss_eval(x, y)
        p = F.softmax(y_hat, dim=1)
        im_loss = entropy(p).mean() + divergence(p)
        return cls_loss * 0.3 + im_loss * 1.0, y_hat

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        self.log_dict({'valid/loss': 1/(self.global_step+1)}, add_dataloader_idx=True)
        return torch.tensor(1/(self.global_step+1), device=batch[0].device)

    def configure_optimizers(self):
        self.fc.requires_grad_(False)

        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.bottleneck.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


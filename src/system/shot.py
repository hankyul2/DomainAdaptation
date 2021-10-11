import os

import torch
from torch import nn
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
        self.make_pseudo_label(nn.Sequential(self.backbone, self.bottleneck), self.fc)

    def make_pseudo_label(self, model, classifier):
        model.eval()
        classifier.eval()
        with torch.no_grad():
            tgt_train = self.trainer.datamodule.train_dataloader()[1].dataset
            tgt_test = self.trainer.datamodule.test_dataloader()
            embed = []
            p = []

            for x, _ in tgt_test:
                embed.append(model(x.to(self.device)))
                p.append(F.softmax(classifier(embed[-1]), dim=1))
            embed, p = torch.cat(embed, dim=0), torch.cat(p, dim=0)

            pseudo_label, centroid = self.cluster(embed, p.t())
            weight = torch.eye(self.num_classes) @ torch.eye(self.num_classes)[pseudo_label].t()
            pseudo_label, centroid = self.cluster(embed, weight.to(self.device))
            tgt_train.samples = [(tgt_train.samples[i][0], pseudo_label[i].item()) for i in range(len(tgt_train))]

        model.train()
        classifier.train()

    def cluster(self, embed, weight):
        centroid = weight @ embed / weight.sum(dim=1, keepdim=True)
        return (F.normalize(embed) @ F.normalize(self.centroid).t()).max(dim=1)[1], centroid

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


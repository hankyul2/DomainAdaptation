from torch import nn

from pytorch_lightning.utilities.cli import instantiate_class

from src.loss.label_smoothing import LabelSmoothing
from src.system.base import BaseVisionSystem


class DABase(BaseVisionSystem):
    def __init__(self, *args, embed_dim: int = 256, dropout: float = 0.1, **kwargs):
        super(DABase, self).__init__(*args, **kwargs)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone.out_channels, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(embed_dim, kwargs['num_classes'])
        self.criterion = LabelSmoothing()

    def forward(self, x):
        return self.fc(self.backbone(self.backbone(x)))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        src, tgt = batch
        return self.shared_step(src, self.train_metric, 'train', add_dataloader_idx=False)

    def compute_loss(self, x, y):
        return self.compute_loss_eval(x, y)

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.bottleneck(self.backbone(x)))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.bottleneck.parameters()},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


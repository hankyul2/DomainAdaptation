from pytorch_lightning.utilities.cli import instantiate_class
from src.system.base import BaseVisionSystem


class Finetune(BaseVisionSystem):
    def __init__(self, *args, **kwargs):
        super(Finetune, self).__init__(*args, **kwargs)

    def compute_loss(self, x, y):
        return self.compute_loss_eval(x, y)

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


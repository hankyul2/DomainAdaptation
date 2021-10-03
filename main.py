from pytorch_lightning import LightningDataModule

from src.cli import MyLightningCLI
from src.system.base import BaseVisionSystem

if __name__ == '__main__':
    cli = MyLightningCLI(BaseVisionSystem, LightningDataModule, save_config_overwrite=True,
                   subclass_mode_data=True, subclass_mode_model=True)
    cli.trainer.test(ckpt_path="best")


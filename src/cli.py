import glob
import os
from typing import Any

import neptune.new as neptune

from torch.optim import Adam, SGD

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.cli import LightningCLI

from src.lr_schedulers import CosineLR, PowerLR


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 1. link argument
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.init_args.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.init_args.max_epochs', apply_on='parse')

        # 2. add optimizer & scheduler argument
        parser.add_optimizer_args((Adam, SGD), link_to='model.init_args.optimizer_init')
        parser.add_lr_scheduler_args((CosineLR, PowerLR), link_to='model.init_args.lr_scheduler_init')

        # 3. add custom argument
        parser.add_argument('--project_name')
        parser.add_argument('--short_id')

        # 4. add shortcut
        self.add_shortcut(parser)

    def add_shortcut(self, parser):
        parser._optionals.conflict_handler = 'resolve'
        shortcut = parser.add_argument_group('Shortcut')
        shortcut.add_argument('-g', '--shortcut.gpus', type=str, help='Enter which gpu you want to use')
        shortcut.add_argument('-e', '--shortcut.max_epochs', type=int, help='Enter the number of epoch')
        shortcut.add_argument('-m', '--shortcut.model_name', type=str.lower,
                            choices=self.get_model_list(), help='Enter model name')
        shortcut.add_argument('-p', '--shortcut.dropout', type=float, help='Enter dropout rate')
        shortcut.add_argument('-d', '--shortcut.dataset_name', type=str, help='Enter dataset')
        shortcut.add_argument('-s', '--shortcut.size', type=int, nargs='+', help='Enter Image Size')
        shortcut.add_argument('-b', '--shortcut.batch_size', type=int, help='Enter batch size for train step')
        shortcut.add_argument('-w', '--shortcut.num_workers', type=int,
                            help='Enter the number of workers per dataloader')
        shortcut.add_argument('-l', '--shortcut.lr', type=float, help='Enter learning rate')

        parser.link_arguments('shortcut.gpus', 'trainer.gpus', apply_on='parse')
        parser.link_arguments('shortcut.gpus', 'model.init_args.gpus', apply_on='parse')
        parser.link_arguments('shortcut.max_epochs', 'trainer.max_epochs', apply_on='parse')
        parser.link_arguments('shortcut.model_name', 'model.init_args.backbone_init.model_name', apply_on='parse')
        parser.link_arguments('shortcut.dropout', 'model.init_args.backbone_init.dropout', apply_on='parse')
        parser.link_arguments('shortcut.dataset_name', 'data.init_args.dataset_name', apply_on='parse')
        parser.link_arguments('shortcut.size', 'data.init_args.size', apply_on='parse')
        parser.link_arguments('shortcut.batch_size', 'data.init_args.batch_size', apply_on='parse')
        parser.link_arguments('shortcut.num_workers', 'data.init_args.num_workers', apply_on='parse')
        parser.link_arguments('shortcut.lr', 'optimizer.init_args.lr', apply_on='parse')

    @staticmethod
    def get_model_list():
        return [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2',
            'mobilenet_v2',
            'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
            'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224',
            'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_384', 'vit_large_patch32_384',
            'r50_vit_base_patch16_224', 'r50_vit_large_patch32_224', 'r50_vit_base_patch16_384',
            'r50_vit_large_patch32_384', 'timm'
        ]

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # 1. load log meta info
        log_dir = 'log'
        config, subcommand = self.get_command_and_config()
        short_id, project_name, dataset_name, model_name = self.get_log_info_from_config(config)
        log_name = self.get_log_name(dataset_name, model_name)
        checkpoint = self.get_checkpoint(log_dir, log_name, short_id, subcommand)

        # 2. define logger
        neptune_logger = self.get_neptune_logger(project_name, log_name, short_id, subcommand)
        neptune_logger.log_hyperparams(config)
        tensorboard_logger = TensorBoardLogger(log_dir, log_name, neptune_logger.version)
        csv_logger = CSVLogger(log_dir, log_name, neptune_logger.version)

        # 3. define callback for Checkpoint, LR Scheduler
        save_dir = os.path.join(log_dir, log_name, neptune_logger.version)
        best_save_dir = os.path.join('pretrained', 'in_this_work', log_name)
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir, save_last=True,
            filename='epoch={epoch}_acc={valid/top@1:.4f}',
            monitor='valid/top@1', mode='max', auto_insert_metric_name=False
        )
        lr_callback = LearningRateMonitor()

        # 4. pass to trainer
        logger = callback = []
        if subcommand == 'fit':
            logger = [neptune_logger, tensorboard_logger, csv_logger]
            callback = [model_checkpoint_callback, lr_callback]
        elif subcommand in ('test', 'predict'):
            logger = [neptune_logger]

        kwargs = {**kwargs, 'logger': logger, 'default_root_dir': save_dir,
                  'callbacks': callback, 'resume_from_checkpoint': checkpoint}

        return super().instantiate_trainer(**kwargs)

    def get_command_and_config(self):
        subcommand = self.config['subcommand']
        config = self.config[subcommand]
        return config, subcommand

    def get_checkpoint(self, log_dir, log_name, short_id, subcommand):
        if short_id:
            best, last = sorted(glob.glob(os.path.join(log_dir, log_name, short_id, '*.ckpt')))[:2]
            if subcommand == 'fit':
                return last
            else:
                self.config_init[subcommand]['ckpt_path'] = best
                return None
        else:
            return None

    @staticmethod
    def get_log_name(dataset_name, model_name):
        return f'{model_name}_{dataset_name}'

    @staticmethod
    def get_log_info_from_config(config):
        short_id = None if config['short_id'] == '' else config['short_id']
        project_name = config['project_name']
        dataset_name = config['data']['init_args']['dataset_name']
        model_name = config['model']['init_args']['backbone_init']['model_name']
        return short_id, project_name, dataset_name, model_name

    @staticmethod
    def get_neptune_logger(project_name, log_name, short_id, subcommand):
        return NeptuneLogger(
                run=neptune.init(
                    project=project_name,
                    api_token=None,
                    name=log_name,
                    run=short_id,
                ),
                prefix=subcommand,
                log_model_checkpoints=False
            )

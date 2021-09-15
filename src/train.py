import torch
from easydict import EasyDict as edict
from torch import nn

from src.model.models import get_model
from src.loss.label_smoothing import LabelSmoothing
from src.base_model_wrapper import BaseModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.log import get_log_name, Result
from src.optimizer import MultiOpt


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, start_time, model, device, criterion, optimizer):
        super().__init__(log_name, start_time)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


def run(args):
    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, shuffle=False, drop_last=False)

    # step 2. load model
    device = torch.device('cuda:{}'.format(args.rank) if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name, nclass=datasets[0].class_num, device=device)

    # step 3. prepare training tool
    criterion = LabelSmoothing()
    optimizer = MultiOpt(model, lr=args.lr, nbatch=len(src_dl), nepoch=args.nepoch)

    # step 4. train
    model = ModelWrapper(log_name=args.log_name, start_time=args.start_time, model=model, device=device, optimizer=optimizer, criterion=criterion)
    model.fit(src_dl, valid_dl, nepoch=args.nepoch)

    # step 5. evaluate
    model.load_best_weight()
    model.evaluate(test_dl)

    # (extra) step 6. save result
    result_saver = Result()
    result_saver.save_result(args, model)


if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
import torch
from easydict import EasyDict as edict

from src.model.models import get_model
from src.loss.label_smoothing import LabelSmoothing
from src.base_model_wrapper import BaseModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.log import get_log_name, Result
from src.optimizer import MultiOpt


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


def run(args):
    # step 1. prepare dataset
    datasets = get_dataset(args.src, args.tgt)
    src_dl, tgt_dl = convert_to_dataloader(datasets[:2], args.batch_size, args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader(datasets[2:], args.batch_size, args.num_workers, train=False)

    # step 2. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name, nclass=datasets[0].class_num).to(device)

    # step 3. prepare training tool
    criterion = LabelSmoothing()
    optimizer = MultiOpt(model, lr=args.lr, nbatch=len(src_dl), nepoch=args.nepoch)

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model=model, device=device, optimizer=optimizer, criterion=criterion)
    model.fit(src_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)


if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
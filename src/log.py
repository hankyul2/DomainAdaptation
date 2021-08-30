import csv
import datetime
import os
import re
import shutil
from glob import glob
from pathlib import Path

from filelock import FileLock


class Result:
    def __init__(self, result_path='log/result.csv', best_result_path='result/best_result'):
        self.result_path = result_path
        self.best_result_save_path = best_result_path
        self.setup()

    @property
    def no(self):
        return 0

    @property
    def acc(self):
        return 5

    @property
    def is_best(self):
        return 10

    @property
    def weight_path(self):
        return 11

    def setup(self):
        Path(self.best_result_save_path).mkdir(exist_ok=True, parents=True)
        if not Path(self.result_path).exists():
            self.write_csv(self.result_path, self.get_headers(), mode='w')

    def read_result(self):
        csv_data = []
        with open(self.result_path, 'r') as f:
            for line in csv.reader(f):
                csv_data.append(line)
        return csv_data, self.csv2dict(csv_data)

    def csv2dict(self, csv_data):
        return {col: [row[c] for row in csv_data[1:]] for c, col in enumerate(csv_data[0])}

    def write_csv(self, result_path, data, mode='w'):
        with open(result_path, mode, newline='') as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)

    def arg2result(self, args, model):
        return [self.get_no(), args.model_name, args.src, args.tgt, model.time,
                model.best_acc.item(), model.best_epoch, args.nepoch, args.lr, args.batch_size, False,
                model.log_best_weight_path]

    def get_no(self):
        csv_list, csv_dict = self.read_result()
        return len(csv_list)

    def check_is_best(self, no):
        csv_list, csv_dict = self.read_result()
        return csv_list[no][self.is_best].lower() == 'true'

    def save_result(self, args, model):
        with FileLock("{}.lock".format(self.result_path)):
            result = self.arg2result(args, model)
            self.write_csv(self.result_path, [result], mode='a')
            best_result, best_idx = self.get_best_result_and_idx(result)
            self.save_best_result(args, best_idx, best_result)

    def save_best_result(self, args, best_idx, best_result):
        if len(best_result) == 3:
            csv_list, csv_dict = self.read_result()
            for no in best_idx:
                csv_list[no][self.is_best] = True
            self.write_csv(self.result_path, csv_list, mode='w')
            self.write_csv('{}/{}.csv'.format(self.best_result_save_path, get_log_name(args).replace('/', '_')),
                           self.get_headers() + best_result)

    def get_headers(self):
        headers = [['no', 'method', 'src', 'tgt', 'start_time', 'acc', 'epoch', 'nepoch',
                    'lr', 'batch_size', 'is_best', 'model_weight_path']]
        return headers

    def get_best_result_and_idx(self, result):
        same_type_result_list = self.get_same_list(result)
        result_len = len(same_type_result_list)
        best_result = best_result_idx = []
        if result_len >= 3:
            valmax = 0
            for i in range(result_len - 3):
                sum_acc = sum([float(row[self.acc]) for row in same_type_result_list[i:i + 3]])
                if sum_acc > valmax:
                    valmax = sum_acc
                    best_result = same_type_result_list[i:i + 3]
                    best_result_idx = [int(row[self.no]) for row in best_result]
        return best_result, best_result_idx

    def get_same_list(self, result, start_end=(1, 5)):
        csv_list, csv_dict = self.read_result()
        return list(filter(lambda xs: all([xs[i] == result[i] for i in range(*start_end)]), csv_list))

    def get_best_model(self, model_name, src, tgt):
        result = [0, model_name, src, tgt]
        models = max(self.get_same_list(result, start_end=(1, 4)),
                     key=lambda xs: float(xs[self.acc]))
        if len(models) < 1:
            assert Exception('Base model are not prepared yet')
        return models[self.weight_path]

    def get_best_model_weight_path(self):
        csv_list, csv_dict = self.read_result()
        return list(map(lambda x: strip_log_name(x[self.weight_path]), filter(lambda xs: xs[self.is_best].lower() == 'true', csv_list)))


def get_base_model(model_name, src, tgt):
    result_saver = Result()
    return result_saver.get_best_model(model_name, src, tgt)


def get_log_name(args, log_format='{}/{}_{}'):
    return log_format.format(args.model_name, args.src, args.tgt)


def is_best_result_log(log_file_path):
    log_name = strip_log_name(log_file_path)
    best_model_path = Result().get_best_model_weight_path()
    return log_name in best_model_path


def strip_log_name(log_file_path):
    return ''.join(log_file_path.split('.')[0].split('/')[2:])


def clear_unused_log(file_path, max_log_len=10):
    log_file_paths = glob(os.path.join(file_path, '*', '*'))
    log_file_paths.sort(key=os.path.getmtime)
    for log_file_path in log_file_paths[:-max_log_len]:
        keep_log = is_best_result_log(log_file_path)
        print("It {} is best result log?  {}".format(log_file_path, 'Yes' if keep_log else 'No'))
        if not keep_log:
            log_file = Path(log_file_path)
            if log_file.exists():
                if log_file.is_dir():
                    shutil.rmtree(log_file)
                else:
                    log_file.unlink(missing_ok=True)


def setup_directory(log_name):
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    for prefix in ['text', 'tensor_board', 'best_weight']:
        Path('log/{}/{}/{}'.format(
            prefix, log_name, date)
        ).mkdir(exist_ok=True, parents=True)
        clear_unused_log('log/{}/{}'.format(prefix, log_name))


if __name__ == '__main__':
    Result()
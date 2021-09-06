import csv
import datetime
import os
import shutil
from glob import glob
from pathlib import Path

from filelock import FileLock
from mdutils.mdutils import MdUtils
from pytz import timezone, utc


def get_log_name(args, log_format='{}/{}_{}'):
    return log_format.format(args.model_name, args.src, args.tgt)


def strip_log_name(log_file_path):
    return ''.join(log_file_path.split('.')[0].split('/')[2:])


class Result:
    def __init__(self, result_path='log/result.csv', best_result_path='result/best_result'):
        self.result_path = result_path
        self.best_result_save_path = best_result_path

        self.headers = ['no', 'model_name', 'src', 'tgt', 'time', 'best_acc', 'best_epoch', 'nepoch',
                        'lr', 'batch_size', 'log_best_weight_path']
        self.best_result_identity_name = 'model_name'
        self.best_result_column_idx = [1, 2, 3]
        self.best_result_save_idx = list(range(len(self.headers) - 1))

        self.best_result_identity_idx = self.headers.index(self.best_result_identity_name)
        self.acc_idx = self.headers.index('best_acc')
        self.weight_path_idx = self.headers.index('log_best_weight_path')
        self.setup_directory()
        self.setup_logfile()

    def setup_directory(self):
        Path(os.path.dirname(self.result_path)).mkdir(exist_ok=True, parents=True)
        Path(self.best_result_save_path).mkdir(exist_ok=True, parents=True)

    def setup_logfile(self):
        if not Path(self.result_path).exists():
            write_csv(self.result_path, [self.headers], mode='w')

    def read_result(self):
        csv_data = []
        with open(self.result_path, 'r') as f:
            for line in csv.reader(f):
                csv_data.append(line)
        return csv_data, csv2dict(csv_data)

    def arg2result(self, args, model):
        result = [self.get_no()]
        for column_name in self.headers[1:]:
            if hasattr(args, column_name):
                object_name = 'args'
            elif hasattr(model, column_name):
                object_name = 'model'
            else:
                raise AssertionError('Args and Model object does not have : {}'.format(column_name))
            exec('result.append({}.{})'.format(object_name, column_name))
        return result

    def get_no(self):
        csv_list, csv_dict = self.read_result()
        return len(csv_list)

    def save_result(self, args, model):
        with FileLock("{}.lock".format(self.result_path)):
            result = self.arg2result(args, model)
            write_csv(self.result_path, [result], mode='a')
            self.update_best_result()

    def update_best_result(self):
        result_list, result_dict = self.read_result()
        for model_name in set(result_dict[self.best_result_identity_name]):
            best_result = []
            for identity_column in self.get_identity_columns(result_list, model_name):
                target_data = self.get_same_list(identity_column, self.best_result_column_idx)
                best_result.append(
                    self.filter_best_result(target_data, num=3 if len(target_data) > 3 else len(target_data)))
            self.make_log_readme(model_name, best_result)

    def get_identity_columns(self, result_list, model_name=None):
        identity_columns = []
        for row in result_list:
            if not model_name or model_name == row[self.best_result_identity_idx]:
                identity_columns.append(
                    tuple(row[dataset_column_idx] for dataset_column_idx in self.best_result_column_idx)
                )
        return set(identity_columns)

    def filter_best_result(self, result, num=3):
        assert len(result) >= num
        best_acc = 0.0
        best_result = result[:num]
        for i in range(len(result) - num):
            acc = sum([float(row[self.acc_idx]) for row in result[i:i + num]])
            if acc > best_acc:
                best_acc = acc
                best_result = result[i:i + num]
        return best_result

    def get_same_list(self, content, mapping):
        csv_list, csv_dict = self.read_result()
        return list(
            filter(lambda row: all([content[org_idx] == row[new_idx] for org_idx, new_idx in enumerate(mapping)]),
                   csv_list))

    def make_log_readme(self, model_name, logs=[]):
        # new readme create
        new_readme = MdUtils(file_name='{}/{}.md'.format(self.best_result_save_path, model_name))
        new_readme.new_line("## {} Summary".format(model_name))
        new_readme.new_line('### Train Log')
        new_readme.new_line('*last modified: {} Time zone is seoul,korea (UTC+9:00)*'.format(get_current_time()))

        for log in logs:
            accs = [float(row[self.acc_idx]) for row in log]
            avg_acc = sum(accs) / len(accs)
            log = [self.headers] + log
            columns, rows = len(self.best_result_save_idx), len(log)
            new_readme.new_paragraph("{} ({:.1f})".format(model_name, avg_acc))
            list_of_strings = [str(row[idx]) for row in log for idx in self.best_result_save_idx]
            new_readme.new_table(columns=columns, rows=rows, text=list_of_strings, text_align='center')
        new_readme.create_md_file()

    def get_best_pretrained_model_path(self, *args):
        assert len(args) == len(self.best_result_column_idx)
        target_data = self.get_same_list(args, self.best_result_column_idx)
        return self.filter_best_result(target_data, num=1)[0][self.weight_path_idx]


def csv2dict(csv_data):
    return {col: [row[c] for row in csv_data[1:]] for c, col in enumerate(csv_data[0])}


def write_csv(result_path, data, mode='w'):
    with open(result_path, mode, newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


def get_current_time():
    KST = timezone(zone='Asia/Seoul')
    now = datetime.datetime.utcnow()
    now_kst_aware = utc.localize(now).astimezone(KST)
    current_time = str(now_kst_aware)
    return current_time


def clear_unused_log(file_path, max_log_len=10):
    log_file_paths = glob(os.path.join(file_path, '*', '*'))
    log_file_paths.sort(key=os.path.getmtime)
    for log_file_path in log_file_paths[:-max_log_len]:
        print("It {} is best result log?  {}".format(log_file_path, 'No'))
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


def run(args):
    Result().update_best_result()

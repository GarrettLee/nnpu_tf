import os
import glob
import time
import pickle
import urllib.request as request
import shutil

import numpy as np

__author__ = 'garrett_local'


def last_modified(path_to_match):
    """
    Get the last modified file among those matching with a given wildcards.
    :param path_to_match: a wildcards indicating what files to find.
    :return: the last modified file among those matching with a given wildcards.
    """
    matched = glob.glob(path_to_match)
    if len(matched) == 0:
        return None
    matched.sort(key=lambda f: os.stat(f).st_mtime)
    newest = matched[-1]
    return newest


def create_dirname_if_not_exist(path):
    create_if_not_exist(os.path.dirname(path))


def create_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):
    with open(file, 'rb') as fo:
        c = pickle.load(fo, encoding='latin1')
    return c


def download(url, des_path='.'):
    try:
        print('downloading {}'.format(url))
        request.urlretrieve(url, os.path.join(des_path,
                                              os.path.basename(url)))
        request.urlcleanup()
    except request.HTTPError as e:
        print('HTTP Error: {} {}'.format(e.code, url))
    except request.URLError as e:
        print('URL Error: {} {}'.format(e.reason, url))


def get_unique_name():
    return '{}'.format(int(time.time()))


def read_exp_log_data(exp_name):
    path = 'result/{}/log/summaries.pkl'.format(exp_name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            log_data = pickle.load(f, encoding='latin1')
    else:
        path = 'result/{}/log/summaries.npz'.format(exp_name)
        d = np.load(path)
        log_data = LogData()
        for key in d.keys():
            setattr(log_data, key, d[key])
    return log_data


def save_basetrings_as_text(basestrings, file_name):
    create_dirname_if_not_exist(file_name)
    with open(file_name, 'wb') as f:
        for s in basestrings:
            f.write(s.encode()+b'\n')


def load_basestrings(file_name):
    with open(file_name, 'r') as f:
        basestrings = f.readlines()
    return [s.strip('\n') for s in basestrings]


def save_log_data(log_data, exp_name):
    path = './result/tmp/{}/log'.format(exp_name)
    create_if_not_exist(path)
    with open(os.path.join(path, 'summaries.pkl'), 'wb') as f:
        pickle.dump(log_data, f, protocol=2)


def settle_saved_data(exp_name):
    shutil.move('./result/tmp/{}'.format(exp_name),
                './result/{}'.format(exp_name),)


class LogData(object):
    def __init__(self):
        self.losses = {}

    def __str__(self):
        return '{}'.format(self.__dict__.keys())

    def log_loss(self, key, loss):
        if key not in self.losses:
            self.losses[key] = [loss]
        else:
            self.losses[key].append(loss)

import os
import tarfile
import numpy as np
import helper.pu_learning_dataset as pu_learning
import helper.file_helper as file_manager

__author__ = 'garrett_local'


def _binarize(labels):
        return ((labels == 0) | (labels == 1) | (labels == 8) |
                (labels == 9)) * 2 - 1


def _prepare_cifar10_data():
    data_path = './data/mldata/'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_manager.create_dirname_if_not_exist(data_path)
    file_name = os.path.basename(url)
    full_path = os.path.join(data_path, file_name)
    folder = os.path.join(data_path, 'cifar-10-batches-py')
    if not os.path.isdir(folder):
        file_manager.download(url, data_path)
        with tarfile.open(full_path) as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=data_path)
    train_x = []
    train_y = []
    for i in range(1, 6):
        file_path = os.path.join(folder, 'data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)

    data_dict = file_manager.unpickle(os.path.join(folder, 'test_batch'))
    test_x = data_dict['data']
    test_y = np.array(data_dict['labels'])

    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32)).\
        transpose([0, 2, 3, 1])
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32)).\
        transpose([0, 2, 3, 1])
    train_y = _binarize(train_y)
    test_y = _binarize(test_y)
    return train_x, train_y, test_x, test_y


class Cifar10Dataset(pu_learning.PuLearningDataSet):

    def __init__(self, cfg):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_cifar10_data()
        super(Cifar10Dataset, self).__init__(cfg)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y


class Cifar10PnDataset(pu_learning.PnLearningDataSet):

    def __init__(self, cfg, prior):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_cifar10_data()
        super(Cifar10PnDataset, self).__init__(cfg, prior)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y

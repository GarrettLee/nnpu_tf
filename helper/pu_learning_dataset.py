import abc
import copy
import math
import numpy as np

import helper.math_helper as math_helper
import helper.cfg_helper as cfg_tools

__author__ = 'garrett_local'


class DataIterator(object):
    def __init__(self, data_lists, batch_size, max_epoch=None, repeat=True,
                 shuffle=True, epoch_finished=None):
        for idx in range(len(data_lists) - 1):
            assert len(data_lists[idx]) == len(data_lists[idx + 1])
        self._data = data_lists
        self._batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._num_data = len(self._data[0])
        assert self._num_data >= self._batch_size
        self._shuffle_indexes = self._maybe_generate_shuffled_indexes()
        self._epoch_finished = 0 if epoch_finished is None else epoch_finished
        self._max_epoch = max_epoch

    @property
    def num_data(self):
        return self._num_data

    @property
    def finished(self):
        if not self._repeat:
            if self.epoch_finished == 1:
                return True
        if self._max_epoch is not None:
            return self.epoch_finished > self._max_epoch
        else:
            return False

    @property
    def epoch_finished(self):
        return self._epoch_finished

    def _maybe_generate_shuffled_indexes(self):
        indexes = list(range(self._num_data))
        if self._shuffle:
            np.random.shuffle(indexes)
        return indexes

    def get_next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        else:
            assert self._num_data >= batch_size
        if len(self._shuffle_indexes) == 0:
            raise StopIteration()
        if len(self._shuffle_indexes) >= batch_size:  # when data left is enough
            indexes = self._shuffle_indexes[:batch_size]
            self._shuffle_indexes = self._shuffle_indexes[batch_size:]
        else:  # when data left is not enough.
            indexes = self._shuffle_indexes
            self._shuffle_indexes = []
        if len(self._shuffle_indexes) == 0:
            self._epoch_finished += 1
            if self._repeat:
                if self._max_epoch is not None:
                    if self._epoch_finished > self._max_epoch:
                        raise StopIteration()
                self._shuffle_indexes = self._maybe_generate_shuffled_indexes()
                num_left = batch_size - len(indexes)
                indexes.extend(self._shuffle_indexes[:num_left])
                self._shuffle_indexes = self._shuffle_indexes[num_left:]
        return [l[indexes] for l in self._data]

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_batch()


class PuDataIterator(object):
    def __init__(self, u_data, l_data, batch_size, max_epoch=None,
                 epoch_finished=0, repeat=True, shuffle=True):
        self._u_num = u_data[0].shape[0]
        self._l_num = l_data[0].shape[0]
        self._data_num = self._u_num + self._l_num
        self._p_u = float(self._u_num) / float(self._u_num + self._l_num)
        self._p_l = float(self._l_num) / float(self._u_num + self._l_num)
        self._batch_size = batch_size
        self._used_u_num, self._used_l_num = 0, 0
        self._u_iterator = DataIterator(u_data, int(batch_size * self._p_u),
                                        repeat=repeat, shuffle=shuffle,
                                        epoch_finished=epoch_finished,
                                        max_epoch=max_epoch)
        self._l_iterator = DataIterator(l_data, int(batch_size * self._p_l),
                                        repeat=repeat, shuffle=shuffle,
                                        epoch_finished=epoch_finished,
                                        max_epoch=max_epoch)
        self._finished_epoch = epoch_finished
        self._max_epoch = max_epoch
        self._repeat = repeat
        self._shuffle = shuffle

    @property
    def epoch_finished(self):
        return self._finished_epoch

    @property
    def num_data(self):
        return self._data_num

    @property
    def finished(self):
        if self._max_epoch is not None:
            return self.epoch_finished > self._max_epoch
        else:
            return False

    def __next__(self):
        used_num = self._used_l_num + self._used_u_num + self._batch_size
        next_u_num = round(used_num * self._p_u - self._used_u_num)
        self._used_u_num += next_u_num
        next_l_num = round(used_num * self._p_l - self._used_l_num)
        next_l_num += self._batch_size - next_u_num - next_l_num
        self._used_l_num += next_l_num
        # Whatever the case, at least one sample is expected from each iterator
        # (though the iterator may be empty, when self._repeat == False).
        assert next_l_num != 0 and next_u_num != 0

        if self._max_epoch is not None:
            if self._finished_epoch >= self._max_epoch:
                raise StopIteration()
        # Stop iteration only if both iterator is finished. So no data will
        # be missed.
        if self._u_iterator.finished and self._l_iterator.finished:
            raise StopIteration()

        try:
            u_data = self._u_iterator.get_next_batch(int(next_u_num))
        except StopIteration:
            u_data = None

        try:
            l_data = self._l_iterator.get_next_batch(int(next_l_num))
        except StopIteration:
            l_data = None

        if not self._repeat:

            # It is guaranteed here that, if one of the iterator is finished,
            # another one will make up the missing part.
            if self._u_iterator.finished and not self._l_iterator.finished:
                u_num = 0 if u_data is None else u_data[0].shape[0]
                left = self._l_iterator.get_next_batch(int(next_u_num - u_num))
                l_data = [np.concatenate((l_data[i], left[i]))
                          for i in range(len(l_data))]
            if self._l_iterator.finished and not self._u_iterator.finished:
                l_num = 0 if l_data is None else l_data[0].shape[0]
                left = self._u_iterator.get_next_batch(int(next_l_num - l_num))
                u_data = [np.concatenate((u_data[i], left[i]))
                          for i in range(len(u_data))]

        self._finished_epoch = min(self._u_iterator.epoch_finished,
                                   self._l_iterator.epoch_finished)
        if u_data is None:
            return l_data
        elif l_data is None:
            return u_data
        else:
            return [np.concatenate((u_data[i], l_data[i]))
                    for i in range(len(l_data))]

    def __iter__(self):
        return self


class PuLearningDataSet(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._num_labeled = int(self._cfg['num_labeled'])
        self.__num_unlabeled = None
        self._overlap = cfg_tools.to_bool(self._cfg['overlap'])
        self._max_epoch = int(self._cfg['max_epoch'])
        self._batch_size = int(self._cfg['batch_size'])
        self._prior = None
        self._shuffled_indexes = None  # shuffle indexes for positive-unlabeled division.
        self._negative = -1
        self._positive = 1

        self._unlabeled_positive_mask = None
        self._unlabeled_negative_mask = None
        self._labeled_positive_mask = None
        self._labeled_negative_mask = None

        # for initialize member variables.
        self._prepare_pu_training_data()

    @property
    def _num_unlabeled(self):
        assert self.__num_unlabeled is not None
        return self.__num_unlabeled

    @property
    def batch_size(self):
        return self._batch_size

    def _prepare_pu_training_data(self):
        positive = self._positive
        negative = self._negative
        train_y = self._original_train_y()
        train_x = self._original_train_x()
        if self._shuffled_indexes is None:
            self._shuffled_indexes = np.array(range(len(train_y)))
            np.random.shuffle(self._shuffled_indexes)
        assert self._shuffled_indexes.shape[0] == len(train_y)
        train_y = train_y[self._shuffled_indexes]
        train_x = train_x[self._shuffled_indexes]
        true_y = copy.deepcopy(train_y)
        num_pos = (train_y == positive).sum()
        num_neg = (train_y == negative).sum()
        if self._overlap:
            self.__num_unlabeled = num_neg + num_pos
            self._prior = float(num_pos) / float(num_neg + num_pos)
            overlapped_indexes = math_helper.mask_to_index(
                train_y == positive
            )[:self._num_labeled]
            train_x_labeled = train_x[overlapped_indexes]
            true_y_labeled = true_y[overlapped_indexes]
            train_x = np.concatenate((train_x, train_x_labeled), axis=0)
            true_y = np.concatenate((true_y, true_y_labeled), axis=0)
            train_y = np.concatenate(
                (negative * np.ones(train_y.shape), np.ones(self._num_labeled)),
            )
        else:
            self.__num_unlabeled = num_neg + num_pos - self._num_labeled
            self._prior = \
                float(num_pos - self._num_labeled) / \
                float(num_neg + num_pos - self._num_labeled)
            train_y[train_y == positive][self._num_labeled:] = negative
        self._unlabeled_positive_mask = np.logical_and(train_y == negative,
                                                       true_y == positive)
        self._unlabeled_negative_mask = np.logical_and(train_y == negative,
                                                       true_y == negative)
        self._labeled_positive_mask = np.logical_and(train_y == positive,
                                                     true_y == positive)
        self._labeled_negative_mask = np.logical_and(train_y == positive,
                                                     true_y == negative)
        return train_x, train_y

    def _prepare_pn_testing_data(self):
        test_y = self._original_test_y()
        test_x = self._original_test_x()
        test_y = test_y.reshape([-1, 1])
        return test_x, test_y

    def get_training_iterator(self, batch_size=None, repeat=True, shuffle=True,
                              max_epoch=None):
        x, y = self._prepare_pu_training_data()
        if max_epoch is None:
            max_epoch = self._max_epoch
        if batch_size is None:
            batch_size = self._batch_size
        y = y.reshape([-1, 1])
        unlabeled_mask = np.logical_or(self._unlabeled_positive_mask,
                                       self._unlabeled_negative_mask)
        u_x = x[unlabeled_mask]
        u_y = y[unlabeled_mask]

        labeled_mask = np.logical_or(self._labeled_positive_mask,
                                     self._labeled_negative_mask)
        l_x = x[labeled_mask]
        l_y = y[labeled_mask]

        return PuDataIterator((u_x, u_y),
                              (l_x, l_y),
                              batch_size, max_epoch=max_epoch, repeat=repeat,
                              shuffle=shuffle)

    def get_testing_iterator(self, batch_size=None):
        x, y = self._prepare_pn_testing_data()
        if batch_size is None:
            batch_size = \
                self._batch_size if self._batch_size <= len(y) else len(y)
        return DataIterator((x, y), batch_size, max_epoch=1, repeat=False,
                            shuffle=False)

    @property
    def prior(self):
        assert self._prior is not None
        return self._prior

    @abc.abstractmethod
    def _original_train_x(self):
        pass

    @abc.abstractmethod
    def _original_train_y(self):
        """
        Should return  binarized labels.
        """
        pass

    @abc.abstractmethod
    def _original_test_x(self):
        pass

    @abc.abstractmethod
    def _original_test_y(self):
        """
        Should return  binarized labels.
        """
        pass


class PnLearningDataSet(object):
    def __init__(self, cfg, prior):
        self._cfg = cfg
        self._max_epoch = int(self._cfg['max_epoch'])
        self._batch_size = int(self._cfg['batch_size'])
        self._prior = prior
        self._shuffled_indexes = None  # shuffle indexes for positive-unlabeled division.
        if prior is None:
            self._num_pos = None
            self._num_neg = None
        else:
            self._num_pos = int(self._cfg['num_pos'])
            self._num_neg = int(math.pow((1 - prior) / (2 * prior), 2) * self._num_pos)

    @property
    def batch_size(self):
        return self._batch_size

    def _prepare_pn_training_data(self):
        positive = 1
        negative = -1
        train_y = self._original_train_y()
        train_x = self._original_train_x()
        if self._shuffled_indexes is None:
            self._shuffled_indexes = list(range(len(train_y)))
            np.random.shuffle(self._shuffled_indexes)
        train_y = train_y[self._shuffled_indexes]
        train_x = train_x[self._shuffled_indexes]
        num_pos = (train_y == positive).sum()
        num_neg = (train_y == negative).sum()
        if self._num_neg is None:
            self._num_neg = num_neg
        if self._num_pos is None:
            self._num_pos = num_pos
        assert num_pos >= self._num_pos and num_neg >= self._num_neg
        train_x = np.concatenate((train_x[train_y == positive][:self._num_pos],
                                 train_x[train_y == negative][:self._num_neg]))
        train_y = np.concatenate((train_y[train_y == positive][:self._num_pos],
                                 train_y[train_y == negative][:self._num_neg]))
        train_y = train_y.reshape([-1, 1])
        return train_x, train_y

    def _prepare_pn_testing_data(self):
        test_y = self._original_test_y()
        test_x = self._original_test_x()
        test_y = test_y.reshape([-1, 1])
        return test_x, test_y

    def get_training_iterator(self, batch_size=None, repeat=True, shuffle=True,
                              max_epoch=None):
        x, y = self._prepare_pn_training_data()
        if max_epoch is None:
            max_epoch = self._max_epoch
        if batch_size is None:
            batch_size = \
                self._batch_size if self._batch_size <= len(y) else len(y)
        return DataIterator((x, y), batch_size, max_epoch=max_epoch,
                            repeat=repeat, shuffle=shuffle)

    def get_testing_iterator(self, batch_size=None):
        x, y = self._prepare_pn_testing_data()
        if batch_size is None:
            batch_size = \
                self._batch_size if self._batch_size <= len(y) else len(y)
        return DataIterator((x, y), batch_size, max_epoch=1, repeat=False,
                            shuffle=False)

    @property
    def prior(self):
        assert self._prior is not None
        return self._prior

    @abc.abstractmethod
    def _original_train_x(self):
        pass

    @abc.abstractmethod
    def _original_train_y(self):
        """
        Should return  binarized labels.
        """
        pass

    @abc.abstractmethod
    def _original_test_x(self):
        pass

    @abc.abstractmethod
    def _original_test_y(self):
        """
        Should return  binarized labels.
        """
        pass

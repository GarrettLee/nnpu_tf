import numpy as np

__author__ = 'garrett_local'


def mask_to_index(mask):
    index = np.array(range(len(mask)))[mask]
    return index


def index_to_mask(index, mask_len):
    mask = np.zeros(mask_len).astype(np.bool)
    mask[index] = True
    return mask


def _crop_common_part(arr1, arr2):
    s1 = arr1.shape
    s2 = arr2.shape
    assert len(s1) == len(s2)
    slc = [slice(0, min(s1[i], s2[i])) for i in range(len(s1))]
    new_mask1 = arr1[slc]
    new_mask2 = arr2[slc]
    return new_mask1, new_mask2


def logical_and(mask1, mask2):
    return np.logical_and(*_crop_common_part(mask1, mask2))


def logical_or(mask1, mask2):
    return np.logical_or(*_crop_common_part(mask1, mask2))


def normalize(arr):
    return arr / np.sum(arr).astype(np.float32)

import numpy as np


def scale(X, feat_range=(0, 1)):
    _min, _max = feat_range

    _X_min, _X_max = X.min(), X.max()

    X_diff = X - _X_min
    X_fraction = (_max - _min) / (_X_max - _X_min)

    X_scaled = X_diff * X_fraction + _min

    return X_scaled


def get_slices(data, slice_size):
    X_length = len(data) - slice_size + 1
    X_shape = (X_length, slice_size)

    X = np.lib.stride_tricks.as_strided(data, X_shape, 2 * data.strides)

    return X


def auto_slice_size(data):
    slice_size = np.ceil(0.03 * len(data))
    slice_size = int(slice_size)

    return slice_size

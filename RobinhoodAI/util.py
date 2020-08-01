import numpy as np
from sklearn.preprocessing import MinMaxScaler


_scaler = MinMaxScaler()


def scale(data):
    return _scaler.fit_transform(data[:, np.newaxis]).squeeze()


def get_slices(data, slice_size, keep_ends=False):
    X_length = len(data) - slice_size
    if keep_ends:
        X_length += 1
    X_shape = (X_length, slice_size)

    X = np.lib.stride_tricks.as_strided(data, X_shape, 2 * data.strides)

    return X

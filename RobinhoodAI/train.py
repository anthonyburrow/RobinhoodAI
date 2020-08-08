import numpy as np
import tensorflow as tf
from tensorflow import keras


from .util import scale, get_slices, auto_slice_size


_max_forecast = 10   # days


def _build_model(X, y):
    model = tf.keras.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True,
                                input_shape=(X.shape[1], 1)))
    model.add(keras.layers.LSTM(units=50, return_sequences=False))
    model.add(keras.layers.Dense(units=25))
    model.add(keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, batch_size=1, epochs=50)

    return model


def generate_model(data, slice_size=None):
    data = np.asarray(data)

    scaled_data = scale(data)

    if slice_size is None:
        slice_size = auto_slice_size(data)

    x_train = get_slices(scaled_data[:-_max_forecast], slice_size=slice_size)
    y_train = get_slices(scaled_data[slice_size:], slice_size=_max_forecast)

    x_train = x_train[:, :, np.newaxis]

    model = _build_model(x_train, y_train)

    return model

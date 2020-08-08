import numpy as np
import tensorflow as tf
from tensorflow import keras


from .util import scale, get_slices


max_forecast = 10   # days


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


def generate_model(X, slice_size):
    X = np.asarray(X)

    X_scaled = scale(X)

    X_train = get_slices(X_scaled[:-max_forecast], slice_size=slice_size)
    y_train = get_slices(X_scaled[slice_size:], slice_size=max_forecast)

    X_train = X_train[:, :, np.newaxis]

    model = _build_model(X_train, y_train)

    return model


def predict(X, model):
    _min, _max = X.min(), X.max()
    X_scaled = scale(X)

    y_predict_scaled = model.predict(X_scaled)

    y_predict = scale(y_predict_scaled, feat_range=(_min, _max))

    return y_predict

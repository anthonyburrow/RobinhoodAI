from numpy import around
import pickle
import os.path

from mypytools.util import log

from RobinhoodAI import _model_path
from read import get_historicals
from train import generate_model
from util import auto_slice_size


class ValidationTest:

    def __init__(self, historical_params, slice_size=None):
        log_fn = 'validation_test'
        self._logger = log.setup_log(log_fn)

        self.data = get_historicals(historical_params)

        self.training_len = around(len(self.data) * 0.8)

        if slice_size is None:
            self._slice_size = auto_slice_size(self.data)
        else:
            self._slice_size = slice_size

    @property
    def model(self):
        model_fn = f'{_model_path}/model_test.p'
        if os.path.isfile(model_fn):
            model = pickle.load(open(model_fn, 'rb'))
        else:
            self._logger.info('Generating model...')
            model = generate_model(data=self.data[:self.training_len],
                                   slice_size=self._slice_size)

            self._logger.info(f'Saving model to {model_fn}...')
            pickle.dump(model, open(model_fn, 'wb'))

        return model

    def run(self):
        y_observed = self.data[self.training_len:]

        y_predicted = self._predict(self.model)

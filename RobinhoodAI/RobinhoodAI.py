import robin_stocks as rh
import sched
import time
import json
import pickle
import os.path

from mypytools.util import log

from read import get_historicals
from train import generate_model
from validation import ValidationTest
from util import auto_slice_size


def_hist_params = {
    'fields': ['high_price'],
    'interval': 'day',
    'span': 'year'
}

_user_path = './user'

_model_path = './models'
_model_fn = 'model.p'


class RobinhoodAI:

    def __init__(self):
        log_fn = 'RobinhoodAI'
        self._logger = log.setup_log(log_fn)

        self._setup()
        self._rh_login()

        self._model = None
        self._slice_size = None

    def _setup(self):
        self._logger.info('Setting up environment...')

        paths = (_user_path, _model_path)
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def _rh_login(self):
        with open(f'{_user_path}/cred.json') as F:
            credentials = json.load(F)

        user = credentials['username']
        pwd = credentials['password']

        self._logger.info('Attempting to log in with Robinhood...')
        rh.login(username=user, password=pwd)
        self._logger.info('Login successful.\n')

    def _rh_logout(self):
        rh.logout()

        self._logger.info('Logged out of Robinhood.\n')

    @property
    def model(self):
        if self._model is None:
            # Load model if available
            model_fn = f'{_model_path}/{_model_fn}'
            if os.path.isfile(model_fn):
                self._model = pickle.load(open(model_fn, 'rb'))
            else:
                msg = ('Model must to be generated with `.train()` before'
                       'attempting to use it.')
                self._logger.error(msg)
                return

        return self._model

    def train(self, stocks, historical_params: dict = def_hist_params,
              slice_size: int = None):
        self._logger.info('Generating model...')

        self._logger.info('Retrieving historical quotes...')
        data = get_historicals(stocks, historical_params)

        if slice_size is None:
            self._slice_size = auto_slice_size(data)
        else:
            self._slice_size = slice_size

        self._logger.info('Generating model...')
        self._model = generate_model(data, self._slice_size)

        # Pickle/save the model after training
        model_fn = f'{_model_path}/{_model_fn}'
        self._logger.info(f'Saving model to {model_fn}...')
        pickle.dump(self._model, open(model_fn, 'wb'))

        self._logger.info('Training complete.\n')

    def _predict(self, X, model):
        pass

    def predict(self):
        pass

    def validation_test(self, stocks,
                        historical_params: dict = def_hist_params,
                        slice_size: int = None):
        self._logger('Performing validation test...')

        val = ValidationTest(stocks, historical_params, slice_size)
        val.run()

        # probably do more with testing/optimization, idk yet

        self._logger('Finished validation test.')

    def start(self):
        self._logger.info('Starting up RobinhoodAI bot...\n')

        # self._scheduler = sched.scheduler(time.time, time.sleep)
        # self._scheduler.enter(1, 1, self._run)
        # self._scheduler.run()

    def stop(self):
        self._rh_logout()

        self._logger.info('Bot stopped successfully.')

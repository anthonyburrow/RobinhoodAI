from pyrh import Robinhood
import numpy as np
import sched
import time
import json
import pickle
import os.path

from mypytools.util import log


# Defaults
historical_fields = 'high_price'
historical_interval = '5minute'
historical_span = 'month'

_user_path = './user'

_model_path = './models'
_model_fn = 'model_full.p'


class RobinhoodAI:

    def __init__(self):
        log_fn = 'RobinhoodAI'
        self._logger = log.setup_log(log_fn)

        self._rh = Robinhood()
        self._rh_login()

        self._model = None

    def _rh_login(self):
        with open(f'{_user_path}/cred.json') as F:
            credentials = json.load(F)

        user = credentials['username']
        pwd = credentials['password']

        self._logger.info('Attempting to log in with Robinhood...')
        self._rh.login(username=user, password=pwd)
        self._logger.info('Login successful.\n')

    def _get_data_single(self, fields, stock, interval, span):
        msg = (f'Retrieving historical quotes for stock {stock} at'
               f'{interval} interval over a {span} span.')
        self._logger.info(msg)

        quotes = self._rh.get_historical_quotes(stock, interval, span)
        quotes = quotes['results'][0]['historicals']

        if isinstance(fields, str):
            data = [date[fields] for date in quotes]
        elif len(fields) == 1:
            f = fields[0]
            data = [date[f] for date in quotes]
        else:
            data = [[date[f] for f in fields] for date in quotes]

        return data

    def _get_data(self, stocks, fields, interval, span):
        if isinstance(stocks, str):
            data = self._get_data_single(fields, stocks, interval, span)
        elif len(stocks) == 1:
            data = self._get_data_single(fields, stocks[0], interval, span)
        else:
            data = [self._get_data_single(fields, s, interval, span)
                    for s in stocks]

        return np.array(data)

    @property
    def model(self):
        if self._model is None:
            # Load model if available
            model_fn = f'{_model_path}/{_model_fn}'
            if os.path.isfile(model_fn):
                model = pickle.load(open(model_fn, 'rb'))
            else:
                msg = ('Model must to be generated with `.train()` before'
                       'attempting to use it.')
                self._logger.error(msg)
                return

        return model

    def _train(self, data, output):
        from .train import generate_model

        self._logger.info('Generating model...')
        model = generate_model(data)

        # Pickle/save the model after training
        model_fn = f'{_model_path}/{output}'
        self._logger.info(f'Saving model to {model_fn}...')
        if not os.path.exists(_model_path):
            os.makedirs(_model_path)
        pickle.dump(model, open(model_fn, 'wb'))

        self._logger.info('Training complete.\n')

        return model

    def train(self, stocks, fields=historical_fields,
              interval=historical_interval, span=historical_span):
        data = self._get_data(stocks, fields, interval, span)

        model_fn = 'model.p'
        self._model = self._train(data, output=model_fn)

    def _predict(self, X):
        pass

    def predict(self):
        pass

    def val_model(self, data):
        # Load model if available
        model_fn = f'{_model_path}/val_model.p'
        if os.path.isfile(model_fn):
            model = pickle.load(open(model_fn, 'rb'))
        else:
            model = self._train(data, output=model_fn)

        return model

    def validation_test(self, stocks, fields=historical_fields,
                        interval=historical_interval, span=historical_span):
        data = self._get_data(stocks, fields, interval, span)

        training_len = np.ceil(len(data) * 0.8)
        model = self.val_model(data[:training_len])

        y_observed = data[training_len:]
        y_predicted = self._predict()
        # do predictions

    def start_bot(self):
        self._logger.info('Starting up RobinhoodAI bot...\n')

        # self._scheduler = sched.scheduler(time.time, time.sleep)
        # self._scheduler.enter(1, 1, self._run)
        # self._scheduler.run()

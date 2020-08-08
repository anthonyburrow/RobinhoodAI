import robin_stocks as rh


def _get_hist_single(stock, fields, interval, span):
    quotes = rh.get_stock_historicals(stock, interval, span)
    quotes = quotes['results'][0]['historicals']

    if isinstance(fields, str):
        data = [date[fields] for date in quotes]
    elif len(fields) == 1:
        f = fields[0]
        data = [date[f] for date in quotes]
    else:
        data = [[date[f] for f in fields] for date in quotes]

    return data


def get_historicals(stocks, historical_params):
    fields = historical_params['fields']
    interval = historical_params['interval']
    span = historical_params['span']

    if isinstance(stocks, str):
        data = _get_hist_single(stocks, fields, interval, span)
    if len(stocks) == 1:
        data = _get_hist_single(stocks[0], fields, interval, span)
    else:
        data = [_get_hist_single(stock, fields, interval, span)
                for stock in stocks]

    return data

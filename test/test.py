from pyrh import Robinhood
from datetime import datetime
import numpy as np
import tulipy as ti
import sched
import time
import json

# username = 'chipperorc'
# password = 'JamesFranco666'

# rh = Robinhood()
# rh.login(username=username, password=password)

# historical_quotes = rh.get_historical_quotes(company_id['ford'], '5minute',
#                                              'day')

# print(historical_quotes)

with open('./user/cred.json') as F:
    credentials = json.load(F)

print(credentials['username'])

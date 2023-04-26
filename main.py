import pandas as pd
import numpy as np
import requests
import beautifulsoup4 as bs4

np.set_printoptions(suppress=True)

from nsepy import get_history
from datetime import date

startDate = date(2019, 1, 1)
endDate = date(2020, 10, 5)

# Fetching the data
StockData = get_history(symbol="SBIN", start=startDate, end=endDate)
print(StockData.shape)
StockData.head()

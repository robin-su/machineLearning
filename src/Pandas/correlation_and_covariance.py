# -*- coding: utf-8 -*- 

import numpy as np
import pandas.io.data as web
from pandas import DataFrame

print('相关性与协方差')  # 协方差：https://zh.wikipedia.org/wiki/%E5%8D%8F%E6%96%B9%E5%B7%AE
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '4/1/2016', '7/15/2015')
    price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
    volume = DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})
returns = price.pct_change() #变化比例
print(returns.tail())
print(returns.MSFT.corr(returns.IBM)) # 微软和IBM股票的相关性
print(returns.corr())  # 相关性，自己和自己的相关性总是1
print(returns.cov()) # 协方差
print(returns.corrwith(returns.IBM)) #计算每个股票跟IBM的相关性
print(returns.corrwith(returns.volume)) # 每个股票和股票成交量的相关性

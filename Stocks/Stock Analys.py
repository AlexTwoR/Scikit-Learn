import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt

import pandas.io.data as web
import datetime

# 1. DATA DOWNLOAD
#
# Fetching Yahoo! for MasterCard Inc. (MA) stock data
start = datetime.datetime(2010, 5, 13)
end = datetime.datetime(2015, 5, 13)


data = web.DataReader("MA", 'yahoo', start, end)['Adj Close']
cp = np.array(data.values) # daily adj-close prices
ret = cp[1:]/cp[:-1] - 1 # compute daily returns

np.mean(ret)
np.std(ret)

# 2. APPLIED STATISTICS
#
# Fit ret with N(mu, sig) distribution and estimate mu and sig
mu_fit, sd_fit = norm.fit(ret)


print(" mu_fit, sig_fit = %.9f, %.9f" % (mu_fit, sd_fit))
dx = 0.001 # resolution
x = np.arange(-5, 5, dx)
pdf = norm.pdf(x, mu_fit, sd_fit)

#VaR

norm.ppf(0.5, mu_fit,sd_fit)

bbStd=pd.rolling_std(data,50)
bbStd
bbUp = data + bbStd
bbDw = data - bbStd

pd.rolling_mean(data,50)
pd.rolling_std(data,50).plot()
bbUp.plot()
bbDw.plot()
data.plot(color='black')
plt.show


data.describe()
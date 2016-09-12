import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#Linear Regression 

from sklearn import linear_model

x = np.arange(20)
x=x.reshape(len(x),1)
x.shape

y = np.arange(20) + 10 + np.random.normal(0,4,20)
y=y.reshape(len(y),1)
y.shape

len(x)==len(y)

plt.plot(x,y)
plt.scatter(x,y)

clf = linear_model.LinearRegression()
clf.fit(x,y)

clf.coef_
clf.intercept_

plt.scatter(x,y)
plt.plot(x, clf.predict(x), color='red')
plt.show()



import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#Linear Regression 

from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Leinear
x = np.arange(20)
x=x.reshape(len(x),1)
x.shape

y = np.arange(20) ** 2 + 10 + np.random.normal(0,50,20)
y=y.reshape(len(y),1)
y.shape

len(x)==len(y)

#plt.plot(x,y)
plt.scatter(x,y)

clf = linear_model.LinearRegression()
clf.fit(x,y)

clf.coef_
clf.intercept_

plt.scatter(x,y)
plt.plot(x, clf.predict(x), color='red')
plt.show()

#Poly
pmodel= make_pipeline(PolynomialFeatures(3), Ridge())
pmodel.fit(x,y)

plt.scatter(x,y)
plt.plot(x, pmodel.predict(x), color='red')
plt.show()



#Support Vector Machine 

from sklearn import svm, datasets

digits=datasets.load_digits()
print(digits.data)

clf = svm.SVC()
clf = svm.SVC(gamma=0.001, C=100)

X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)

digits.target[-4]
print(clf.predict(digits.data[-4]))




#From Habr
#https://habrahabr.ru/company/mlclass/blog/247751/

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import urllib

# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]

X.shape
y.shape


from sklearn import preprocessing

#Нормализация и Стандартизация
X_norm = preprocessing.normalize(X)
X_stan = preprocessing.scale(X)
print(X)
print(X_norm)
print(X_stan)


#


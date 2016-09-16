from sklearn import datasets
%pylab inline

import matplotlib.pyplot as plt


def plot_2d_dataset(data, colors):
    pyplot.figure(figsize(8, 8))
    pyplot.scatter(map(lambda x: x[0], data[0]), map(lambda x: x[1], data[0]), c = data[1], cmap = colors)

#---- datasets.make_circles -----

circles = datasets.make_circles(100, shuffle=True, factor=0.75, noise=0.1)

print "features: {}".format(circles[0][:10])
print "target: {}".format(circles[1][:10])

from matplotlib.colors import ListedColormap

colors = ListedColormap(['r','b','y'])
pyplot.figure(figsize(8, 8))
plt.scatter(map(lambda x: x[0], circles[0]), map(lambda x: x[1], circles[0]),   
                c = circles[1], cmap = colors)


#---- datasets.make_classification -----

#n_redundant - corr features
X1, Y1 = datasets.make_classification(n_features=4, n_redundant=1, n_informative=1, n_clusters_per_class=1)

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

import pandas as pd
cox = pd.DataFrame(X1)
cox.corr()


#---- datasets.make_blobs -----

blobs = datasets.make_blobs(centers = 4, cluster_std = 1.5, random_state=1)

pyplot.Figure(figsize(8,8))
pyplot.scatter(map(lambda x:x[0],blobs[0]), map(lambda x:x[1],blobs[0]), c=blobs[1])



#---- datasets.make_regression -----

data, target, coef = datasets.make_regression(n_features = 2, n_informative = 1, n_targets = 1, 
                                              noise = 5.0, coef = True, random_state = 2)
                                              
                                              
pylab.scatter(map(lambda x:x[0], data), target, color = 'r')
pylab.scatter(map(lambda x:x[1], data), target, color = 'b')

pylab.scatter(data[:,0], target, color = 'r')
pylab.scatter(data[:,1], target, color = 'b')



#---- datasets.load_iris ----
iris = datasets.load_iris()

iris.keys()

print "feature names: {}".format(iris.feature_names)
print "target names: {names}".format(names = iris.target_names)



#---- IRIS ----
iris.data[1:10]
iris.target

from pandas import DataFrame

iris_frame = DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame['target'] = iris.target

print(iris_frame)



#---- seaborn ----
#https://stanford.edu/~mwaskom/software/seaborn/index.html
import numpy as mp
import seaborn as sns

sns.pairplot(iris_frame, hue = 'target')
sns.corr(iris_frame)

sns.heatmap(iris_frame.corr())

iris_frame.corr()

# Cool tool!

X1, Y1 = datasets.make_classification(n_features=5, n_redundant=2, 
                                    n_informative=1, n_clusters_per_class=1)

cox = DataFrame(X1)
sns.heatmap(cox.corr())
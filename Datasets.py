from sklearn import datasets
%pylab inline

import matplotlib.pyplot as plt



#---- datasets.make_circles -----

circles = datasets.make_circles(100, shuffle=True, factor=0.75, noise=0.1)

print "features: {}".format(circles[0][:10])
print "target: {}".format(circles[1][:10])

from matplotlib.colors import ListedColormap

colors = ListedColormap(['r','b'])
pyplot.figure(figsize(8, 8))
plt.scatter(map(lambda x: x[0], circles[0]), map(lambda x: x[1], circles[0]),   
                c = circles[1], cmap = colors)
                
                
                

#---- datasets.make_classification -----

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import cross_validation, datasets, linear_model, metrics


X1, Y1 = datasets.make_classification(n_features=5, n_redundant=2, n_informative=3, 
                                        n_clusters_per_class=1, random_state=1)

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)



# Correlation exam
x_fr=pd.DataFrame(X1)
x_fr
sns.heatmap(x_fr.corr())


# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:19:37 2020

@author: Adi
"""

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,3:5].values

# using the dendogram for optimal number of clustering 
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = "ward"))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# fitting hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters= 5,affinity= 'euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

# visualising the clusters 
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s= 100, c='red', label= 'Careful')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s= 100, c='magenta', label= 'Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s= 100, c='cyan', label= 'Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s= 100, c='green', label= 'Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s= 100, c='blue', label= 'Sensible')
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
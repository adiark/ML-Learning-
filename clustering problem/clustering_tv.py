# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:41:35 2020

@author: Adi
"""

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# importing dataset
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:,0:13].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# using the dendogram for optimal number of clustering 
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = "ward"))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# fitting hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters= 3,affinity= 'euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

# visualising the clusters 
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s= 100, c='red', label= 'Type 1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s= 100, c='magenta', label= 'Type 2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s= 100, c='cyan', label= 'Type 3')
plt.title("Clusters of Wines")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend()
plt.show()
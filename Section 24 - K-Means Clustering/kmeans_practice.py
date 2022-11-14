# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:50:39 2020

@author: Adi
"""
# importing libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# importing data set 
dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values

# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss= []
for i in range(1,11):
    kmeans= KMeans(n_clusters= i,init= "k-means++",n_init=10,max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show

# applying kmeans to the mall dataset
kmeans= KMeans(n_clusters= 5,init= "k-means++",n_init=10,max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# visualising the clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s= 100, c='red', label= 'Careful')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s= 100, c='magenta', label= 'Standard')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s= 100, c='cyan', label= 'Target')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s= 100, c='green', label= 'Careless')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s= 100, c='blue', label= 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 300, c= 'yellow', label = 'Centroids')
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
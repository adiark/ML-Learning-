# hierarchical clustering 

# Importing the dataset

dataset = read.csv("Mall_Customers.csv")
X = dataset[,4:5]

# using dendogram to find optimal number of clusters
dendogram = hclust(d = dist(X, method = "euclidean"), method = "ward.D")
plot(dendogram,
     main = "Dendogram",
     xlab = "Customers",
     ylab = "Euclidean distance")

# fitting clustering to the dataset 
hc = hclust(d = dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc, k = 5)

# visualising the clusters
library(cluster)
clusplot(x = X,y_hc,lines = 0, shade = TRUE, color =  TRUE, labels = 2, plotchar = FALSE, span = TRUE, 
         main= "Clusters of Clients", xlab= "Annual Income", ylab= "Spending Score")

# K means clustering 

# imorting dataset
dataset= read.csv("Mall_Customers.csv")
X <- dataset[,4:5]

# using elbow method to find optimal number of clusters
set.seed(5)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
plot(1:10, wcss,type = 'b',main = "Clusters of Clients",xlab = "Number of Clusters", ylab = "WCSS")

# applying k means for mall dataset
kmeans <- kmeans(X,5,iter.max = 300,nstart = 10)

# visualising the cluster
library(cluster)
clusplot(x = X,kmeans$cluster,lines = 0, shade = TRUE, color =  TRUE, labels = 2, plotchar = FALSE, span = TRUE, 
         main= "Clusters of Clients", xlab= "Annual Income", ylab= "Spending Score")

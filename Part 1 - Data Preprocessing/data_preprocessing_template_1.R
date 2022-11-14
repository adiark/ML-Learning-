setwd("C:\\Users\\Adi\\Desktop\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing")

dataset <- read.csv("Data.csv")

# taking care of the missing data 

dataset$Age <- ifelse(test = is.na(dataset$Age), 
                      ave(x = dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Age)
dataset$Salary <- ifelse(test = is.na(dataset$Salary), 
                      ave(x = dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Salary)

# encoding categorical variables

dataset$Country <- factor(x = dataset$Country, levels = c("France", "Spain", "Germany"), labels = c(1,2,3))
dataset$Purchased <- factor(x = dataset$Purchased, levels = c("No", "Yes"), labels = c(0,1))

# divide data into train and test set 
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set <- subset(dataset, split== TRUE)
test_set <- subset(dataset, split== FALSE)

# feature scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:09:13 2020

@author: Adi
"""

# import library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset= pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_x= LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder= OneHotEncoder(categorical_features=[0])
x= onehotencoder.fit_transform(x).toarray()


labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the data into train and test set 
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state= 0)

# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
x_train= sc_X.fit_transform(x_train)
x_test= sc_X.transform(x_test)
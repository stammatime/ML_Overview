# -*- coding: utf-8 -*-
"""
Created on 1/28/18

@author: Cody
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

# Importing the dataset
# : is all the lines
# :-1 is all the columns except last one
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 3].values

# Handle the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy="mean", axis=0)
#index 1 to 2 (upper bound not computed)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
# Converting category to a number
X[:,0] = labelEncoder_X.fit_transform(X[:, 0])

# Now we convert each category to its own column: france, germany, and spain
# 1 means is, 0 means is not
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Only one category so we can use label encoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# ---------------------
# Because salary and age are both numbers, their values need to be scaled to each other
# otherwise, age 44 may have a ton less weight than 61000 salary
# We need to make sure euclidean distance between the two points is to scale
# Salary Euclidean distance between 79000 and 48000 is (3100)^2=961000000
# Age Euclidean distance between 48 and 27 is (21)^2=441
# Difference between these two is so large that age will be dominated by salary

# Feature scaling types
# ---------------------
# Standardization => (x-mean(x))/stdDeviation(x)
# Normalisation => (x-min(x))/(max(x)-min(x))
# ---------------------

from sklearn.preprocessing import StandardScaler
# fit object to training set and then transform it
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # sc_X already fit for training set, so we do not need to do twice

# We don't need to apply feature scaling to the dependent variables (purchased) because range is
# simply 0 to 1




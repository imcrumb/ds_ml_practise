# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:37:06 2018

@author: Ian Lynch
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
#print(dataset.describe(include='all',))
#print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
#Encoding State

#LabelEncoder() encodes values with labels between 0 and n_classes-1
labelencoder_state = LabelEncoder()
X[:, 3] = labelencoder_state.fit_transform(X[:, 3])

#OneHotEncoder() encodes categorical int features using one-hot scheme
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting multiple linear regression to the training set X_train, y_train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting test set results 
y_pred = regressor.predict(X_test)

#building the optimal model using bwd elimination
#we need to add a column of ones to our predictors because OLS model does not
#include intercept
#recall "building a model" slides

#1 Select significance level, say SL = 0.05
#2 Fit the full model with all possible predictors
import statsmodels.formula.api as sm
ones = np.ones(shape=(50,1))
X = np.append(arr = ones,values =X, axis =1)

#mae = mean_absolute_error(y_test,y_pred)
#print(mae)

#3 Consider the predictor with the highest p-value. If p > SL goto step #4 else FIN
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# px2 > SL so remove px2

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# px1 > SL so remove px1

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# px2 > SL so remove px2

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# px2 > SL so remove px2

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

y_opt = regressor_OLS.predict(X_opt)

#review below get true comparison
"""
mae = mean_absolute_error(y_test,y_pred)
print(mae)
mae = mean_absolute_error(y,y_opt)
print(mae)
""" 
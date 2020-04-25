# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:06:38 2020

@author: Shrikant Agrawal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
df = pd.read_csv('50_Startups.csv')

# Split depnedent and independent variable
x=df.iloc[:,:-1]
y=df.iloc[:,4]

# Convert categorical varible into numerical by using onehot encoading
state=pd.get_dummies(x['State'],drop_first=True)

""" drop_frist = True means - we have 3 caterories NY, Cal, FL. 
For NY    Cal    FL
    0     0      1  = FL
    1     0      0  = NY
    0     1      0  = Cal
    
    when we get 0 in Cal and 0 in Fl which means record belongs from NY.
    So we can remove column first and hence we used this command.
    IT is call as DUMMY VARIABLE TRAP."""

# State column is no more required in x variable. axis = 1 means column
x=x.drop('State', axis=1)    

# now combine x and state data using concate
x=pd.concat([x,state],axis=1)

#Now split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Fitting Multiple Linear regression to the training dataset. We have same library for
#simple linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set results
y_pred= regressor.predict(x_test)

#Measure the accuracy of our model using R^2 value (r-square)

""" Rsquare value always ranges between 0 - 1. If it is more closer to 1 means our model is 
good, most of the time for good model fit it ranges between 0.8-1

R square = 1- SSres/SSmean       SSres= sum of resuduals and SSmean = sum of mean

SSres= 1/n Summation(y-y^)^2        y^ is y hat = predicted values of y

SSmean= 1/n Summation(y-y-)^2        y- is y mean = mean values of y

Hence, SSmean always greater than SSres, hence we get very small number of R2 value"""

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score

# R squared value can also be calculated as below
score1 = regressor.score(x_test, y_test)
score1

#Our model has given us 93% accuracy, means model is good


# we can not check accuracy using confusion matrix because its continuous variable
from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test, y_pred)
accuracy



















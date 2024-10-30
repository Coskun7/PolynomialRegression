#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:35:19 2024

@author: mali
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/Users/mali/Downloads/Ice_cream selling data.csv')

x=df.iloc[:,0].to_numpy()
y=df.iloc[:,1].to_numpy()
x_train = x.reshape(-1,1)
y_train = y.reshape(-1,1)




plt.scatter(x,y)
plt.xlabel('Temperature (°C)');plt.ylabel('Ice Cream Sales (units)')
plt.show()

plr = PolynomialFeatures(degree=2)
lr = LinearRegression()

lr.fit(x_train,y_train)
y_pred=lr.predict(x_train)

x_poly=plr.fit_transform(x_train)
lr.fit(x_poly,y_train)
y_head=lr.predict(x_poly)


plt.plot(x_train,y_head,label='Poly',color='r')
plt.plot(x_train,y_pred,label='Linear',color='g')
plt.scatter(x_train,y)
plt.xlabel('Temperature (°C)');plt.ylabel('Ice Cream Sales (units)')
plt.show()





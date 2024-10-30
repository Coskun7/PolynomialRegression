#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:58:22 2024

@author: mali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/Users/mali/Downloads/polynomial+regression.csv',sep=';')


x=df.iloc[:,0].to_numpy()
y=df.iloc[:,1].to_numpy()

x_train=x.reshape(-1,1)
y_train=y.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel('Araba Fiyatı')
plt.ylabel('Araba Hızı')
plt.show()

lin_model = LinearRegression()

lin_model.fit(x_train,y_train)

y_pred = lin_model.predict(x_train)


plt.plot(x,y_pred,color='red')
plt.scatter(x,y)
plt.xlabel('Araba Fiyatı')
plt.ylabel('Araba Hızı')
plt.show()

lin_model.predict([[10000]])

from sklearn.preprocessing import PolynomialFeatures
lrp = PolynomialFeatures(degree=2)

x_polynomial = lrp.fit_transform(x_train)


lin_model.fit(x_polynomial,y_train)

y_head = lin_model.predict(x_polynomial)

plt.plot(x_train,y_head,color='green',label='polynomial')
plt.plot(x,y_pred,color='red',label='simple')
plt.scatter(x,y)
plt.xlabel('Araba Fiyatı')
plt.ylabel('Araba Hızı')
plt.legend()
plt.show()







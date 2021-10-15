# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:34:39 2021

@author: bob
"""
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

df = pd.read_csv("FuelConsumption.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color='b')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly,train_y)

print('Coefficients: ',clf.coef_)
print('Intercept: ', clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )

poly_3 = PolynomialFeatures(degree=3)
train_x_poly2 = poly_3.fit_transform(train_x)

clf2 = linear_model.LinearRegression()
train_y_2 = clf2.fit(train_x_poly2,train_y)

print('Coefficient: ',clf2.coef_)
print('Intercept: ', clf2.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='b')
yy2 = clf2.intercept_[0] + clf2.coef_[0][1]*XX + clf2.coef_[0][2]*np.power(XX,2) + clf2.coef_[0][3]*np.power(XX,3)
plt.plot(XX,yy2,'-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

test_x_poly2 = poly_3.fit_transform(test_x)
test_y_2 = clf2.predict(test_x_poly2)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_2 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_2 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_2 ) )
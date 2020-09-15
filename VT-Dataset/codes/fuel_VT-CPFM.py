#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the coefficient
of determination are also calculated.

"""
# print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import KFold
from keras.layers import Dense, Reshape, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import math
# from dataset_ems_read import data_access_ems

def data_access():

	file_vehicle_1 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"
	# file_vehicle_2 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"
	# file_vehicle_3 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"

	df_data = pd.read_csv(file_vehicle_1,index_col=False,usecols=[0,1,2,3,4,5,6,7,8],header=0,names=['CO2', 'CO', 'HC', 'NOx', 'vel','fuel', 'engine', 'elevation', 'phase_num'])

	print(df_data)

	X = []
	y = []

	for i in range(len(df_data) - 1):
		CO2 = float(df_data['CO2'][i])
		CO = float(df_data['CO'][i])
		HC = float(df_data['HC'][i])
		NOx = float(df_data['NOx'][i])
		vel = float(df_data['vel'][i])
		engine = float(df_data['engine'][i])
		elevation = float(df_data['elevation'][i])
		phase_num = float(df_data['phase_num'][i])

		# data_features = [CO2,CO,HC,NOx,vel,engine,elevation,phase_num]
		data_features = [CO2,CO,HC,NOx,vel,engine]

		fuel = float(df_data['fuel'][i])
		data_label = [fuel]

		X.append(data_features)
		y.append(data_label)

	return X,y


def kfold_load(X,y):
	n_splits = 5
	kfold = KFold(n_splits=n_splits, shuffle=False)

	print('KFold = %d :'%n_splits)
	for train_index, test_index in kfold.split(X, y):
	    train_X, test_X = X[train_index], X[test_index]
	    train_y, test_y = y[train_index], y[test_index]
	    yield train_X, train_y, test_X, test_y

# X,y = data_access()
X,y = data_access()
y = np.squeeze(y)

X = np.array(X)
Y = np.array(y)

r = 0
acc = 0

for train_X, train_y, test_X, test_y in kfold_load(X,y):
	print("\n")
	print(len(train_X))
	n = 5
	poly = PolynomialFeatures(n)# returns: [1, x, x^2, x^3]
	# print(poly.fit_transform([[7]]))
	trX_expanded = poly.fit_transform(train_X)
	# print(trX_expanded[0])
	teX_expanded = poly.fit_transform(test_X)

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(trX_expanded, train_y)

	# Make predictions using the testing set
	y_pred = regr.predict(teX_expanded)

	# The coefficients
	# print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print('R Mean squared error: %.2f' % math.sqrt(mean_squared_error(test_y, y_pred)))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f' % r2_score(test_y, y_pred))
	r += r2_score(test_y, y_pred)

	print('median_absolute_error: %.2f' % median_absolute_error(test_y, y_pred))
	print('mean value: %.2f' % np.mean(test_y))
	accuracy = 1 - median_absolute_error(test_y, y_pred) / np.mean(test_y)
	print('average accuracy: %.2f' % accuracy)

	acc += accuracy

print(r/5)
print(acc/5)
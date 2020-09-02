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

def data_access_ems():
	file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
	file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"
	# file_fuel_rate = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x721_Ins_flow_rate.csv"
	file_ems_fuel = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x18fef2fe_EngFuelRate.csv"

	# file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
	# file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"
	# file_fuel_rate = "/Users/torres_kai/Downloads/fuel-dataset-3/0x721_Ins_flow_rate.csv"

	df_ems_rate = pd.read_csv(file_ems_fuel,index_col=False,header=None,sep='	')
	# print(df_ems_rate)

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	# df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,23,24,25,26,27,33,37],names=['time','throttle_position_percent',
	# 	'engine_torque_percent','driver_demand_engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','combined_vehicle_weight_kg', 'vehicle_speed_mps'])
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,33,37,39,40],names=['time'
		,'combined_vehicle_weight_kg','vehicle_speed_mps','longitudinal_acceleration_mpss','lateral_acceleration_mpss'])
	# df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,24,26,27,30],names=['time',
	# 	'engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','cur_gear_pos'])
	df_rate = pd.read_csv(file_ems_fuel,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])
	# df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])

	# print(df_rate)
	# print(len(df_vehicle['time']))
	t0 = 1590556664.10647
	j = 1
	n = 0

	X = []
	y = []

	for i in range(len(df_rate['time'])):
	# for i in range(90298,842000):
		t = t0 + df_rate['time'][i]
		if j >= len(df_vehicle['time']) - 1:
    			break
		while j < 858400 and float(df_vehicle['time'][j])/1000000000 < t:
    			j = j + 1

		vehicle_speed_mps = float(df_vehicle['vehicle_speed_mps'][j-1])
		# brake_position_percent = float(df_vehicle['brake_position_percent'][j-1])
		# retarder_actual_torque_percent = float(df_vehicle['retarder_actual_torque_percent'][j-1])
		# clutch_slip_rate_percent = float(df_vehicle['clutch_slip_rate_percent'][j-1])
		combined_vehicle_weight_kg = float(df_vehicle['combined_vehicle_weight_kg'][j-1])
		# throttle_position_percent = float(df_vehicle['throttle_position_percent'][j-1])
		# engine_torque_percent = float(df_vehicle['engine_torque_percent'][j-1])
		# driver_demand_engine_torque_percent = float(df_vehicle['driver_demand_engine_torque_percent'][j-1])
		# engine_torque_loss_percent = float(df_vehicle['engine_torque_loss_percent'][j-1])
		# engine_speed_rpm = float(df_vehicle['engine_speed_rpm'][j-1])
		# combined_vehicle_weight_kg = float(df_vehicle['combined_vehicle_weight_kg'][j-1])
		# vehicle_speed_mps = float(df_vehicle['vehicle_speed_mps'][j-1])
		# cur_gear_pos = float(df_vehicle['cur_gear_pos'][j-1])
		lateral_acceleration_mpss = float(df_vehicle['lateral_acceleration_mpss'][j-1])

		fuel_rate = float(df_rate['fuel_rate'][i]) * 10

		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,vehicle_speed_mps,brake_position_percent,
		# retarder_actual_torque_percent,clutch_slip_rate_percent,combined_vehicle_weight_kg]
		# data_ems = [engine_speed_rpm,engine_torque_percent,cur_gear_pos,retarder_actual_torque_percent]
		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos]
		data_ems = [vehicle_speed_mps,lateral_acceleration_mpss]
		# print(data_ems)
		data_label = [fuel_rate]

		# if fuel_rate == 0:
		# 	continue

		if j < 90298 or j > 825000:
			continue
		# print(data_ems)
		# print(data_label)

		# if vehicle_speed_mps < 11:
		# 	continue
		# print(vehicle_speed_mps)

		X.append(data_ems)
		y.append(data_label)

		# print(i)
		# print(n)

		# print(t, float(df_vehicle['time'][j-1])/1000000000, t - float(df_vehicle['time'][j-1])/1000000000)

		n = n + 1

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
X,y = data_access_ems()
y = np.squeeze(y)

# print(X)
# for i in range(len(X)):
# 	print(i)
# 	plt.scatter(X[i][0],X[i][1])
# plt.show()

X = np.array(X)
Y = np.array(y)
# print(X.shape)

# Split the data into training/testing sets
# X_train = X[:-10000]
# X_test = X[-10000:]
# print(len(X_train))
# print(len(X_test))

# Split the targets into training/testing sets
# y_train = y[:-10000]
# y_test = y[-10000:]
# print(len(y_train))
# print(len(y_test))

r = 0
acc = 0

for train_X, train_y, test_X, test_y in kfold_load(X,y):
	n = 7
	poly = PolynomialFeatures(n)# returns: [1, x, x^2, x^3]
	trX_expanded = poly.fit_transform(train_X)
	# print(train_X[0])
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

	# plt.plot(test_y[:1000])
	# plt.plot(y_pred[:1000],color="red")
	# plt.show()
	# inp = Input((n+1)) 
	# #since one of the features is 1, we need an extra input
	# out = Dense(1)(inp)
	# model = Model(inputs=inp, outputs=out)
	# model.compile(optimizer=Adam(lr=1e-3), loss="mean_squared_error")

	# model.fit(trX_expanded, train_y, epochs=100)

	# y_pred = model.predict(teX_expanded)

	# # The coefficients
	# # print('Coefficients: \n', regr.coef_)
	# # The mean squared error
	# print('Mean squared error: %.2f' % mean_squared_error(test_y, y_pred))
	# # The coefficient of determination: 1 is perfect prediction
	# print('Coefficient of determination: %.2f' % r2_score(test_y, y_pred))
	# r += r2_score(test_y, y_pred)

print(r/5)
print(acc/5)

# Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

def data_access():
	file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
	file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"
	file_fuel_rate = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x721_Ins_flow_rate.csv"

	# file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
	# file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"
	# file_fuel_rate = "/Users/torres_kai/Downloads/fuel-dataset-3/0x721_Ins_flow_rate.csv"

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	# df_location = pd.read_csv(file_localization,index_col=False,header=0)
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,23,24,25,26,27,33,37],names=['time','throttle_position_percent',
		'engine_torque_percent','driver_demand_engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','combined_vehicle_weight_kg', 'vehicle_speed_mps'])
	# df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	')
	df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])

	print(df_rate)
	print(len(df_vehicle['time']))
	t0 = 1590556664.10647
	j = 1
	n = 0

	X = []
	y = []

	for i in range(len(df_rate['time'])):
	# for i in range(10):
		t = t0 + df_rate['time'][i]
		if j >= len(df_vehicle['time']) - 1:
    			break
		while j < len(df_vehicle['time']) and float(df_vehicle['time'][j])/1000000000 < t:
    			j = j + 1
		print(j)
		print(n)

		throttle_position_percent = float(df_vehicle['throttle_position_percent'][j-1])
		engine_torque_percent = float(df_vehicle['engine_torque_percent'][j-1])
		driver_demand_engine_torque_percent = float(df_vehicle['driver_demand_engine_torque_percent'][j-1])
		engine_torque_loss_percent = float(df_vehicle['engine_torque_loss_percent'][j-1])
		engine_speed_rpm = float(df_vehicle['engine_speed_rpm'][j-1])
		combined_vehicle_weight_kg = float(df_vehicle['combined_vehicle_weight_kg'][j-1])
		vehicle_speed_mps = float(df_vehicle['vehicle_speed_mps'][j-1])

		fuel_rate = float(df_rate['fuel_rate'][i])

		data_ems = [throttle_position_percent,engine_torque_percent,driver_demand_engine_torque_percent,engine_torque_loss_percent,engine_speed_rpm, combined_vehicle_weight_kg, vehicle_speed_mps]
		data_label = [fuel_rate]

		X.append(data_ems)
		y.append(data_label)

		print(type(throttle_position_percent))

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

X,y = data_access()
y = np.squeeze(y)

X = np.array(X)
Y = np.array(y)

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

for train_X, train_y, test_X, test_y in kfold_load(X,y):
	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(train_X, train_y)

	# Make predictions using the testing set
	y_pred = regr.predict(test_X)

	# The coefficients
	print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print('Mean squared error: %.2f' % mean_squared_error(test_y, y_pred))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f' % r2_score(test_y, y_pred))
	r += r2_score(test_y, y_pred)

print(r/5)
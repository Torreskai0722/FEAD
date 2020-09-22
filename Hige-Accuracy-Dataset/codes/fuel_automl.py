#!/usr/bin/python
# -*- coding: utf-8 -*-

# Regression Example With Boston Dataset: Baseline
import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score
from pandas import read_csv,read_excel
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import math
import time
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import autosklearn.classification
import autosklearn.regression

look_back = 4

def data_access():
	file_localization = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/localization_result.csv"
	file_vehicle_report = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/vehicle_report.csv"
	file_fuel_rate = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/0x721_Ins_flow_rate.csv"
	file_ems_fuel = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/0x18fef2fe_EngFuelRate.csv"

	# df_ems_rate = pd.read_csv(file_ems_fuel,index_col=False,header=None,sep='	')
	# print(df_ems_rate)

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	# df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,23,24,25,26,27,33,37],names=['time','throttle_position_percent',
	# 	'engine_torque_percent','driver_demand_engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','combined_vehicle_weight_kg', 'vehicle_speed_mps'])
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,9,10,24,26,27,30,32,33,37,39,40],names=['time','brake_position_percent',
		'retarder_actual_torque_percent','engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','cur_gear_pos',
		'clutch_slip_rate_percent','combined_vehicle_weight_kg','vehicle_speed_mps','longitudinal_acceleration_mpss','lateral_acceleration_mpss'])
	# df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,24,26,27,30],names=['time',
	# 	'engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','cur_gear_pos'])
	# df_rate = pd.read_csv(file_ems_fuel,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])
	df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])

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
		brake_position_percent = float(df_vehicle['brake_position_percent'][j-1])
		retarder_actual_torque_percent = float(df_vehicle['retarder_actual_torque_percent'][j-1])
		clutch_slip_rate_percent = float(df_vehicle['clutch_slip_rate_percent'][j-1])
		combined_vehicle_weight_kg = float(df_vehicle['combined_vehicle_weight_kg'][j-1])
		# throttle_position_percent = float(df_vehicle['throttle_position_percent'][j-1])
		engine_torque_percent = float(df_vehicle['engine_torque_percent'][j-1])
		# driver_demand_engine_torque_percent = float(df_vehicle['driver_demand_engine_torque_percent'][j-1])
		engine_torque_loss_percent = float(df_vehicle['engine_torque_loss_percent'][j-1])
		engine_speed_rpm = float(df_vehicle['engine_speed_rpm'][j-1])
		# combined_vehicle_weight_kg = float(df_vehicle['combined_vehicle_weight_kg'][j-1])
		# vehicle_speed_mps = float(df_vehicle['vehicle_speed_mps'][j-1])
		cur_gear_pos = float(df_vehicle['cur_gear_pos'][j-1])
		longitudinal_acceleration_mpss = float(df_vehicle['longitudinal_acceleration_mpss'][j-1])
		lateral_acceleration_mpss = float(df_vehicle['lateral_acceleration_mpss'][j-1])

		fuel_rate = float(df_rate['fuel_rate'][i])
		# fuel_rate = float(df_rate['fuel_rate'][i]) * 10

		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,vehicle_speed_mps,brake_position_percent,
		# retarder_actual_torque_percent,clutch_slip_rate_percent,combined_vehicle_weight_kg,lateral_acceleration_mpss]
		# data_ems = [engine_speed_rpm,engine_torque_percent,cur_gear_pos,retarder_actual_torque_percent]
		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos]
		data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,lateral_acceleration_mpss,longitudinal_acceleration_mpss]
		# data_ems = [vehicle_speed_mps,lateral_acceleration_mpss,longitudinal_acceleration_mpss]
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


X, y = data_access()

y = np.squeeze(y)
X = np.array(X)
y = np.array(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# automl = autosklearn.classification.AutoSklearnClassifier()

r = 0
acc = 0

#automl = autosklearn.regression.AutoSklearnRegressor()

for X_train, y_train, X_test, y_test in kfold_load(X,y):

	automl = autosklearn.regression.AutoSklearnRegressor()
	automl.fit(X_train, y_train)
	y_hat = automl.predict(X_test)

	r2 = r2_score(y_test, y_hat)
	print("R2 score", r2)
	accuracy = 1 - median_absolute_error(y_test, y_hat) / np.mean(y_test)
	print("Accuracy score", accuracy)
	# print(automl.get_models_with_weights())
	
	r += r2
	acc += accuracy

print(r/5)
print(acc/5)

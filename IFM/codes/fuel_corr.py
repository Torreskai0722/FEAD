#!/usr/bin/python
# -*- coding: utf-8 -*-

# Regression Example With Boston Dataset: Baseline
import pandas as pd
import pymysql
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv,read_excel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sb
import math

def data_access():
	file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
	file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"
	file_fuel_rate = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x721_Ins_flow_rate.csv"

	# file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
	# file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"
	# file_fuel_rate = "/Users/torres_kai/Downloads/fuel-dataset-3/0x721_Ins_flow_rate.csv"

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	# df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,23,24,25,26,27,33,37],names=['time','throttle_position_percent',
	# 	'engine_torque_percent','driver_demand_engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','combined_vehicle_weight_kg', 'vehicle_speed_mps'])
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,9,10,24,26,27,30,32,33,37],names=['time','brake_position_percent',
		'retarder_actual_torque_percent','engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','cur_gear_pos',
		'clutch_slip_rate_percent','combined_vehicle_weight_kg','vehicle_speed_mps'])
	# df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,24,26,27,30],names=['time',
	# 	'engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','cur_gear_pos'])
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

		fuel_rate = float(df_rate['fuel_rate'][i])

		data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,vehicle_speed_mps,brake_position_percent,
		retarder_actual_torque_percent,clutch_slip_rate_percent,combined_vehicle_weight_kg]
		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos]
		data_label = [fuel_rate]

		# if fuel_rate == 0:
		# 	continue

		if j < 90298:
			continue

		# if vehicle_speed_mps < 11:
		# 	continue
		# print(vehicle_speed_mps)

		X.append(data_ems)
		y.append(data_label)

		print(i)
		# print(n)

		# print(t, float(df_vehicle['time'][j-1])/1000000000, t - float(df_vehicle['time'][j-1])/1000000000)

		n = n + 1

	return X,y


X,Y = data_access()
print(X)
scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

# train_size = int(len(X)*0.6)
# test_size = len(X) - train_size
# train, test = X[0:train_size,:],X[train_size:len(X),:]


Z = np.hstack((X,Y))

#C_mat = np.cov(Z)
C_mat = np.corrcoef(Z.T)
print(C_mat.shape)
print(C_mat)
fig = plt.figure(figsize = (10,10))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()
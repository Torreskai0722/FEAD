#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Regression Example With Boston Dataset: Baseline
import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from pandas import read_csv,read_excel
from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import math
import time
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
# from dataset_ems_read import data_access_ems

look_back = 10

def data_access_ems():
	file_localization = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/localization_result.csv"
	file_vehicle_report = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/vehicle_report.csv"
	file_fuel_rate = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/0x721_Ins_flow_rate.csv"
	file_ems_fuel = "/home/nvidia/projects/FEAD/Hige-Accuracy-Dataset/0x18fef2fe_EngFuelRate.csv"

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,9,10,24,26,27,30,32,33,37,39,40],names=['time','brake_position_percent',
		'retarder_actual_torque_percent','engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','cur_gear_pos',
		'clutch_slip_rate_percent','combined_vehicle_weight_kg','vehicle_speed_mps','longitudinal_acceleration_mpss','lateral_acceleration_mpss'])
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

		# fuel_rate = float(df_rate['fuel_rate'][i]) * 10
		fuel_rate = float(df_rate['fuel_rate'][i])

		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,vehicle_speed_mps,brake_position_percent,
		# retarder_actual_torque_percent,clutch_slip_rate_percent,combined_vehicle_weight_kg,lateral_acceleration_mpss]
		# data_ems = [engine_speed_rpm,engine_torque_percent,cur_gear_pos,retarder_actual_torque_percent]
		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos]
		data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,lateral_acceleration_mpss,longitudinal_acceleration_mpss]
		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,lateral_acceleration_mpss]
		# data_ems = [vehicle_speed_mps,lateral_acceleration_mpss,longitudinal_acceleration_mpss]
		data_label = [fuel_rate]

		# if fuel_rate == 0:
		# 	continue

		if j < 90298 or j > 825000:
			continue

		X.append(data_ems)
		y.append(data_label)

		n = n + 1

	return X,y

# define wider model
def cnn_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=6,kernel_initializer='normal', activation='relu'))
	# model.add(Dense(100, input_dim=3*look_back,kernel_initializer='normal', activation='relu'))
	# model.add(Dropout(0.2))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Reshape((5,5,4)))
	model.add(TimeDistributed(Convolution1D(128, 4),input_shape=(4,5,2)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Flatten())
	model.add(Dense(1))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

X,Y = data_access_ems()

scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)
# print(X)
# print(Y)

X = np.array(X)
Y = np.array(Y)

# checkpoint
filepath="mlp-weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

trainX, trainY, testX, testY = train_test_split(X, y, random_state=1)

model = cnn_model()

history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.1,callbacks=callbacks_list, verbose=1)

np.savetxt('cnn_loss.csv', history.history['loss'])
np.savetxt('cnn_val_loss.csv', history.history['val_loss'])

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
trainr2 = r2_score(trainY, trainPredict)
print('Train Score: %.2f RMSE' % (trainScore))
print(trainr2)
testScore = math.sqrt(mean_squared_error(testY, testPredict))
testr2 = r2_score(testY, testPredict)
print('Test Score: %.2f RMSE' % (testScore))
print(testr2)

print('median_absolute_error: %.2f' % median_absolute_error(testY, testPredict))
print('mean value: %.2f' % np.mean(testY))
accuracy = 1 - median_absolute_error(testY, testPredict) / np.mean(testY)
print('average accuracy: %.2f' % accuracy)

t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
model.save('model/cnn/CNN_model_%s+%f.h5'%(t0,testr2))

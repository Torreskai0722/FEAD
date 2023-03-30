#!/usr/bin/python
# -*- coding: utf-8 -*-

# Regression Example With Boston Dataset: Baseline
import pandas as pd
import pymysql
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from pandas import read_csv,read_excel
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sb
import math
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
import time
from sklearn.preprocessing import PolynomialFeatures
# from dataset_ems_read import data_access_ems

look_back = 5
epochs = 50

def data_access_ems():
	file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
	file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"
	file_fuel_rate = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x721_Ins_flow_rate.csv"
	file_ems_fuel = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x18fef2fe_EngFuelRate.csv"

	# file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
	# file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"
	# file_fuel_rate = "/Users/torres_kai/Downloads/fuel-dataset-3/0x721_Ins_flow_rate.csv"

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

		# fuel_rate = float(df_rate['fuel_rate'][i]) * 10
		fuel_rate = float(df_rate['fuel_rate'][i])
		# print(df_rate['fuel_rate'][i])

		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,vehicle_speed_mps,brake_position_percent,
		# retarder_actual_torque_percent,clutch_slip_rate_percent,combined_vehicle_weight_kg]
		# data_ems = [engine_speed_rpm,engine_torque_percent,cur_gear_pos,retarder_actual_torque_percent]
		# data_ems = [engine_speed_rpm,engine_torque_percent,engine_torque_loss_percent,cur_gear_pos,lateral_acceleration_mpss]
		data_ems = [vehicle_speed_mps,lateral_acceleration_mpss,longitudinal_acceleration_mpss]
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


# define the model
def lstm_model(trainX,trainY,testX,testY,scaler):
	model = Sequential()
	model.add(Dense(100, input_dim=3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Reshape((5,5,4)))
	model.add(TimeDistributed(Convolution1D(128, 4)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	# model.add(Flatten())
	model.add(Dense(200, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal'))
	# create and fit the LSTM network
	model.add(LSTM(300,dropout=0.2,return_sequences=True))
	model.add(LSTM(300,dropout=0.2,return_sequences=True))
	model.add(LSTM(200,dropout=0.2,return_sequences=True))
	model.add(LSTM(50,return_sequences=False))
	# model.add(Flatten())
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')

	# checkpoint
	filepath="cnnlstm-adam-weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	model.fit(trainX, trainY, epochs=epochs, batch_size=64,validation_split=0.1, verbose=1,callbacks=callbacks_list)
	# model.fit(trainX, trainY, epochs=epochs, batch_size=20, verbose=1)

	# evaluate larger model
	# estimators = []
	# estimators.append(('standardize', StandardScaler()))
	# estimators.append(('lstm', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=70, verbose=1)))
	# pipeline = Pipeline(estimators)
	# kfold = KFold(n_splits=10)
	# results = np.sqrt(-cross_val_score(pipeline, X, Y, cv=kfold,scoring='neg_mean_squared_error')).mean()
	# r2 = cross_val_score(pipeline, X, Y, cv=kfold,scoring='r2').mean()
	# print(results)
	# print(r2)


	model.load_weights("cnnlstm-adam-weights.best.hdf5")
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	trainr2 = r2_score(trainY[0], trainPredict[:,0])
	print('Train Score: %.2f RMSE' % (trainScore))
	print(trainr2)
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	testr2 = r2_score(testY[0], testPredict[:,0])
	print('Test Score: %.2f RMSE' % (testScore))
	print(testr2)

	print('median_absolute_error: %.2f' % median_absolute_error(testY[0], testPredict[:,0]))
	print('mean value: %.2f' % np.mean(testY[0]))
	accuracy = 1 - median_absolute_error(testY[0], testPredict[:,0]) / np.mean(testY[0])
	print('average accuracy: %.2f' % accuracy)

	t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
	model.save('model/cnnlstm/CNNLSTM_model_%s+%f.h5'%(t0,testr2))
	# model.summary()

	return testr2,accuracy


# X,Y = data_access()
X,Y = data_access_ems()
print("load data success")
scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)

X = np.array(X)
Y = np.array(Y)

# n = 5
# poly = PolynomialFeatures(n)# returns: [1, x, x^2, x^3]
# X = poly.fit_transform(X)
# print(X.shape)
# print(X[0])

r = 0
acc = 0

for train_X, train_y, test_X, test_y in kfold_load(X,Y):
	# trainX, trainY = [],[]
	# for i in range(len(train_X)-look_back-1):
	# 	a = train_X[i:(i+look_back),:]
	# 	trainX.append(a)
	# 	trainY.append(train_y[i+look_back])

	# testX, testY = [],[]
	# for i in range(len(test_X)-look_back-1):
	# 	a = test_X[i:(i+look_back),:]
	# 	testX.append(a)
	# 	testY.append(test_y[i+look_back])
	# print(np.array(train_X).shape)
	# trainX = np.reshape(train_X,(np.array(train_X).shape[0],2,2,1))
	# testX = np.reshape(test_X,(np.array(test_X).shape[0],2,2,1))
	# print(trainX.shape)
	# lstm_model(trainX,trainY,testX,testY,scaler)
	r2, accuracy = lstm_model(train_X,train_y,test_X,test_y,scaler)
	r += r2
	acc += accuracy

print("CNNLSTM G3 without")
print(r/5)
print(acc/5)

# train_size = int(len(X)*0.8)
# test_size = len(X) - train_size
# train, test = X[0:train_size,:],X[train_size:len(X),:]
# Y_train, Y_test = Y[0:train_size],Y[train_size:len(Y)]

# model.save('models/LSTM_model_%f_epoch-%d.hdf5'%(testr2,epochs))

# estimators.append(('lstm', KerasRegressor(build_fn=lstm_model, epochs=100, batch_size=50, verbose=1)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = np.sqrt(-cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')).mean()
# r2 = cross_val_score(pipeline, X, Y, cv=kfold, scoring='r2').mean()
# print('LSTM model:')
# print(results)
# print(r2)
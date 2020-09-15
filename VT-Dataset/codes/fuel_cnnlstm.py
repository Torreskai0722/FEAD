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

		# print(i)

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
	model.add(Dense(100, input_dim=6, kernel_initializer='normal', activation='relu'))
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
X,Y = data_access()
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

print("CNNLSTM G6")
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
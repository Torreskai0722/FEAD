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
from keras.layers import Dense
from keras.layers import LSTM, Reshape
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sb
import math
from keras.callbacks import ModelCheckpoint
import time
from keras.utils import plot_model
from sklearn.preprocessing import PolynomialFeatures
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
# from dataset_ems_read import data_access_ems

look_back = 10
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
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(100, input_shape=(look_back,6),dropout=0.2,return_sequences=True))
	model.add(LSTM(100,return_sequences=True,dropout=0.2))
	model.add(LSTM(100,return_sequences=True,dropout=0.2))
	model.add(LSTM(100,return_sequences=True,dropout=0.2))
	model.add(LSTM(100,return_sequences=True,dropout=0.2))
	# model.add(Reshape((5,4,4)))
	# model.add(TimeDistributed(Convolution1D(128, 4)))
	# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	# model.add(TimeDistributed(Flatten()))
	model.add(LSTM(50,return_sequences=False))
	# model.add(Dense(40, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(10, kernel_initializer='normal'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')

	# checkpoint
	filepath="adam-weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	history = model.fit(trainX, trainY, epochs=epochs, batch_size=32,validation_split=0.1, verbose=1,callbacks=callbacks_list)

	np.savetxt('lstm_loss.csv', history.history['loss'])
	np.savetxt('lstm_val_loss.csv', history.history['val_loss'])

	# Plot training & validation loss values
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('Model loss')
	# plt.ylabel('Loss')
	# plt.xlabel('Epoch')
	# plt.legend(['Train', 'Test'], loc='upper left')
	# plt.show()

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


	# model.load_weights("adam-weights.best.hdf5")
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
	model.save('model/lstm/LSTM_model_%s+%f.h5'%(t0,testr2))
	plot_model(model, to_file='lstm-model.png')
	model.summary()

	return testr2,accuracy


X,Y = data_access()
print("LSTM G6")
print("load data success")
X = np.array(X)
print(np.var(X))
print(np.mean(X))

# Plot training & validation loss values
# plt.plot(X)
# plt.plot(Y)
# plt.title('Fuel Consumption Analysis')
# plt.ylabel('Fule rate')
# plt.xlabel('Time')
# plt.legend(['X', 'Y'], loc='upper right')
# plt.show()

scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)

X = np.array(X)
Y = np.array(Y)
# print(X[0:5])

r = 0
acc = 0

for train_X, train_y, test_X, test_y in kfold_load(X,Y):
	trainX, trainY = [],[]
	for i in range(len(train_X)-look_back-1):
		a = train_X[i:(i+look_back),:]
		trainX.append(a)
		trainY.append(train_y[i+look_back])

	testX, testY = [],[]
	for i in range(len(test_X)-look_back-1):
		a = test_X[i:(i+look_back),:]
		testX.append(a)
		testY.append(test_y[i+look_back])

	print(np.array(trainX).shape)
	trainX = np.reshape(trainX,(np.array(trainX).shape[0],look_back,np.array(trainX).shape[2]))
	# print(trainX[1],trainX[2])
	# print(trainY[1],trainY[2])
	testX = np.reshape(testX,(np.array(testX).shape[0],look_back,np.array(testX).shape[2]))
	r2,accuracy = lstm_model(trainX,trainY,testX,testY,scaler)
	r += r2
	acc += accuracy

print(r/5)
print(acc/5)
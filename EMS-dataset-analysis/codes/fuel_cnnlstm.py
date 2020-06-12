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

look_back = 5
epochs = 100

def load_emsdata():
	## 加上字符集参数，防止中文乱码
	dbconn=pymysql.connect(
	  host="localhost",
	  database="fuel_efficiency",
	  user="root",
	  password="Liu0722*",
	  port=3306,
	  charset='utf8'
	 )

	#sql语句
	sqlcmd="SELECT ems_value,lat,lng FROM emsdata_2 where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '重庆市') AND ((tdate = '20200405') or (tdate = '20200411') or (tdate = '20200412') or (tdate = '20200417'))"
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"

	#利用pandas 模块导入mysql数据
	a=pd.read_sql(sqlcmd,dbconn)
	#取前5行数据

	X = []
	y = []

	b = a
	# b = a.head()
	tfc = 0
	for i in range(len(b.index)):
		# print(i)
		d = json.loads(b['ems_value'][i])
		try:
			# data_label = [float(d['x7001'])]
			label = float(d['x7001'])
			engine = float(d['x7000'])
			throttle = float(d['x7006'])
			speed = float(d['x006C'])
			torque = float(d['x704F'])
		except KeyError:
			continue
		# data_ems = [float(d.get('x7006', '0')),float(d['x7001'])]
		# data_ems = [float(d.get('x7000', '0')),float(d['x7001']),float(d.get('x7006', '0')),float(d.get('x006C', '0'))]
		# data_ems = [float(d.get('x7000', '0')),float(d['x7001']),float(d.get('x7006', '0')),float(d.get('x006C', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# try:
		# 	data_label = [float(d['x7002'])]
		# 	tfc = d['x7002']
		# except KeyError:
		# 	data_label = [float(d.get('x7002',tfc))]
		# print(data_ems)
		data_ems = [engine,throttle]
		data_label = [label]
		print(data_ems)
		X.append(data_ems)
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


# define the model
def lstm_model(trainX,trainY,testX,testY,scaler):
	model = Sequential()
	model.add(Dense(100, input_dim=21, kernel_initializer='normal', activation='relu'))
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

	t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
	model.save('model/cnnlstm/CNNLSTM_model_%s+%f.h5'%(t0,testr2))
	# model.summary()


X,Y = load_emsdata()
print("load data success")
scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)

X = np.array(X)
Y = np.array(Y)

n = 5
poly = PolynomialFeatures(n)# returns: [1, x, x^2, x^3]
X = poly.fit_transform(X)
print(X.shape)
print(X[0])

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
	lstm_model(train_X,train_y,test_X,test_y,scaler)

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
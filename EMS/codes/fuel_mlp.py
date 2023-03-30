#!/usr/bin/python
# -*- coding: utf-8 -*-

# Regression Example With Boston Dataset: Baseline
import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv,read_excel
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import math
import time
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

look_back = 4

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
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01') AND (tdate = '20191201') AND (city = '怀化市')"
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"
	sqlcmd="SELECT ems_value,lat,lng FROM emsdata_2 where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '重庆市') AND ((tdate = '20200405') or (tdate = '20200411') or (tdate = '20200412') or (tdate = '20200417'))"
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata_2 where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '重庆市')"
	# sqlcmd="SELECT ems_value,torque FROM emsdata_2 where (truckid = '107624CE6B76B627818179C74FF206CF') AND (city = '衡阳市') AND ((tdate = '20200403') or (tdate = '20200406') or (tdate = '20200410') or (tdate = '20200413') or (tdate = '20200416') or (tdate = '20200419'))"
	# sqlcmd="SELECT ems_value,torque FROM emsdata_2 where (truckid = '107624CE6B76B627818179C74FF206CF') AND (city = '湘潭市') AND ((tdate = '20200404') or (tdate = '20200408') or (tdate = '20200412') or (tdate = '20200414') or (tdate = '20200417'))"
	# sqlcmd="SELECT ems_value,torque FROM emsdata_2 where (truckid = '107EF28DD023DD7F50AEDA9CA033325B') AND (city = '荆州市') AND ((tdate = '20200403') or (tdate = '20200407') or (tdate = '20200415') or (tdate = '20200419'))"
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (county = '青羊区')"

	#利用pandas 模块导入mysql数据
	a=pd.read_sql(sqlcmd,dbconn)
	#取前5行数据

	X = []
	y = []

	b = a
	# b = a.head()
	tfc = 0
	for i in range(len(b.index)):
		print(i)
		d = json.loads(b['ems_value'][i])
		try:
			label = float(d['x7001'])
			engine = float(d['x7000'])
			throttle = float(d['x7006'])
			speed = float(d['x006C'])
			torque = float(d['x704F'])
		except KeyError:
			continue
		# data_ems = [float(d.get('x7006', '0'))]
		# print(d)
		# data_ems = [float(d.get('x7000', '0')),float(d.get('x7006', '0')),float(d.get('x006C', '0'))]
		# data_ems = [float(d.get('x7000', '0')),float(d.get('x7006', '0')),float(d.get('x7007', '0')),float(d.get('x006C', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# try:
		# 	data_label = [float(d['x7002'])]
		# 	tfc = d['x7002']
		# except KeyError:
		# 	data_label = [float(d.get('x7002',tfc))]
		# data_ems = [engine,torque]
		data_ems = [engine,torque]
		data_label = [label]
		print(data_ems,label)
		X.append(data_ems)
		y.append(data_label)

	return X,y

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(11, input_dim=9, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define wider model
def cnn_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=56,kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Reshape((5,5,4)))
	model.add(TimeDistributed(Convolution1D(128, 4)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(Flatten())
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def kfold_load(X,y):
	n_splits = 5
	kfold = KFold(n_splits=n_splits, shuffle=False)

	print('KFold = %d :'%n_splits)
	for train_index, test_index in kfold.split(X, y):
	    train_X, test_X = X[train_index], X[test_index]
	    train_y, test_y = y[train_index], y[test_index]
	    yield train_X, train_y, test_X, test_y

# load dataset
# dataframe = read_excel("housing.xlsx", delim_whitespace=True, header=None)
# dataset = dataframe.values
# split into input (X) and output (Y) variables
# X = dataset[:,0:13]
# Y = dataset[:,13]
X,Y = load_emsdata()
# print(X)
# print(Y)
print(np.mean(Y))
print(np.var(Y))

scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)

X = np.array(X)
Y = np.array(Y)

# checkpoint
filepath="mlp-weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

r = 0

for train_X, train_y, test_X, test_y in kfold_load(X,Y):
	# n = 5
	# poly = PolynomialFeatures(n)# returns: [1, x, x^2, x^3]
	# print(poly)
	# train_X = poly.fit_transform(train_X)
	# print(train_X[0])
	# print(trX_expanded[0])
	# test_X = poly.fit_transform(test_X)

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

	# trainX = np.reshape(trainX,(np.array(trainX).shape[0],np.array(trainX).shape[2],5,8))
	# testX = np.reshape(testX,(np.array(testX).shape[0],np.array(testX).shape[2],5,8))

	model = larger_model()
	# model = cnn_model()
	history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_split=0.2,callbacks=callbacks_list, verbose=1)

	np.savetxt('mlp_loss.csv', history.history['loss'])
	np.savetxt('mlp_val_loss.csv', history.history['val_loss'])

	trainPredict = model.predict(train_X)
	testPredict = model.predict(test_X)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([train_y])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([test_y])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	trainr2 = r2_score(trainY[0], trainPredict[:,0])
	print('Train Score: %.2f RMSE' % (trainScore))
	print(trainr2)
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	testr2 = r2_score(testY[0], testPredict[:,0])
	print('Test Score: %.2f RMSE' % (testScore))
	print(testr2)
	r += testr2

	t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
	# model.save('model/mlp/MLP_model_%s.h5'%t0)
	model.save('model/mlp/MLP_model_%s+%f.h5'%(t0,testr2))
	model.summary()
	
print(r/5)

# scores = model.evaluate(X, Y, verbose=0)
# print(model.metrics_names[0])
# print(str(scores))
# print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))

# evaluate model
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=20, verbose=1)
# kfold = KFold(n_splits=10)
# results = np.sqrt(-cross_val_score(estimator, X, Y, cv=kfold,scoring='neg_mean_squared_error')).mean()
# r2 = cross_val_score(estimator, X, Y, cv=kfold,scoring='r2').mean()
# print(results)
# print(r2)

# evaluate model with standardized dataset
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=20, verbose=1)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = np.sqrt(-cross_val_score(pipeline, X, Y, cv=kfold,scoring='neg_mean_squared_error')).mean()
# r2 = cross_val_score(pipeline, X, Y, cv=kfold,scoring='r2').mean()
# print(results)
# print(r2)

# evaluate larger model
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=20, verbose=1)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = np.sqrt(-cross_val_score(pipeline, X, Y, cv=kfold,scoring='neg_mean_squared_error')).mean()
# r2 = cross_val_score(pipeline, X, Y, cv=kfold,scoring='r2').mean()
# print(results)
# print(r2)

# evaluate wider model
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = np.sqrt(-cross_val_score(pipeline, X, Y, cv=kfold,scoring='neg_mean_squared_error')).mean()
# r2 = cross_val_score(pipeline, X, Y, cv=kfold,scoring='r2').mean()
# print(results)
# print(r2)
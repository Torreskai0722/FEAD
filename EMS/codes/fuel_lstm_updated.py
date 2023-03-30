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
from keras.callbacks import ModelCheckpoint
import time
from keras.utils import plot_model
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
from sklearn.svm import SVR

look_back = 10
epochs = 500

def load_emsdata(da):
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
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"
	sqlcmd="SELECT ems_value,lat,lng,triggertime FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND (tdate = %s)" % da
	#利用pandas 模块导入mysql数据
	a=pd.read_sql(sqlcmd,dbconn)

	X = []
	y = []

	b = a
	tfc = 0
	for i in range(len(b.index)):
		print('+++++++++++++++')
		print(i)
		d = json.loads(b['ems_value'][i])
		try:
			# data_label = [float(d['x7001'])]
			label = float(d['x7001'])
			engine = float(d['x7000'])
			throttle = float(d['x7006'])
			speed = float(d['x006C'])
		except KeyError:
			continue
		# data_ems = [float(d['x7001'])]
		# data_ems = [float(d.get('x7000', '0')),float(d['x7001']),float(d.get('x7006', '0')),float(d.get('x006C', '0'))]
		# data_ems = [float(d.get('x7000', '0')),float(d['x7001']),float(d.get('x7006', '0')),float(d.get('x006C', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# data_ems = [float(d.get('x7000', '0')),float(d.get('x7003', '0')),float(d.get('x7004', '0')), 
		# float(d.get('x7005', '0')),float(d.get('x7006', '0')),float(d.get('x7007', '0')),float(d.get('x006C', '0')),float(d.get('x7035', '0')),float(d.get('x7091', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# data_ems = [float(d.get('x7000', '0')),float(d['x7001']),float(d.get('x7006', '0')),float(d.get('x006C', '0'))]
		# try:
		# 	data_label = [float(d['x7002'])]
		# 	tfc = d['x7002']
		# except KeyError:
		# 	data_label = [float(d.get('x7002',tfc))]
		# print(data_ems)
		data_ems = [engine,throttle,speed,label]
		data_label = [label]
		# print(b['triggertime'][i])
		# print(d['x000B'])
		X.append(data_ems)
		y.append(data_label)

	return X,y

# def kfold_load(X,y):
# 	n_splits = 5
# 	kfold = KFold(n_splits=n_splits, shuffle=False)

# 	print('KFold = %d :'%n_splits)
# 	for train_index, test_index in kfold.split(X, y):
# 	    train_X, test_X = X[train_index], X[test_index]
# 	    train_y, test_y = y[train_index], y[test_index]
# 	    yield train_X, train_y, test_X, test_y


# define the model
def lstm_model(trainX,trainY,testX,testY,input_scaler):
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(100, input_shape=(look_back,4),dropout=0.2,return_sequences=True))
	model.add(LSTM(100,return_sequences=True,dropout=0.2))
	model.add(LSTM(100,return_sequences=True,dropout=0.2))
	model.add(LSTM(50,return_sequences=True))
	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal'))
	model.add(Flatten())
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')

	history = model.fit(trainX, trainY, epochs=epochs, batch_size=32, verbose=1)

	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = input_scaler.inverse_transform(trainPredict)
	trainY = input_scaler.inverse_transform([trainY])
	testPredict = input_scaler.inverse_transform(testPredict)
	testY = input_scaler.inverse_transform([testY])
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
	model.save('model/lstm/LSTM_model_%s+%f.h5'%(t0,testr2))
	plot_model(model, to_file='lstm-model.png')
	# model.summary()

train_X,train_y = load_emsdata('20191204')
test_X, test_y = load_emsdata('20191209')

input_scaler =  StandardScaler()
# output_scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
train_X = input_scaler.fit_transform(train_X)
train_y = input_scaler.fit_transform(train_y)
test_X = input_scaler.fit_transform(test_X)
test_y = input_scaler.fit_transform(test_y)

train_y = np.squeeze(train_y)
test_y = np.squeeze(test_y)
train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)

print("load data success")

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

# print(np.array(trainX).shape)
trainX = np.reshape(trainX,(np.array(trainX).shape[0],look_back,np.array(trainX).shape[2]))
testX = np.reshape(testX,(np.array(testX).shape[0],look_back,np.array(testX).shape[2]))
lstm_model(trainX,trainY,testX,testY,input_scaler)
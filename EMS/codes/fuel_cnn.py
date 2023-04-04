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
from keras.layers import Dense, Reshape, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
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

look_back = 10

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
	# sqlcmd="SELECT ems_value FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01') AND (tdate = '20191201') AND (city = '怀化市')"
	# sqlcmd="SELECT ems_value FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"
	sqlcmd="SELECT ems_value FROM emsdata_2 where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '重庆市') AND ((tdate = '20200405') or (tdate = '20200411') or (tdate = '20200412') or (tdate = '20200417'))"
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
			torque = float(d['x704F'])
			# speed = float(d['x006C'])
		except KeyError:
			continue
		# data_ems = [float(d.get('x7005', '0')),float(d['x7001'])]
		# data_ems = [float(d.get('x7091', '0')),float(d['x7002']),float(d.get('x7005', '0'))]
		# data_ems = [float(d.get('x7000', '0')),float(d['x7002']),float(d.get('x7005', '0')),float(d.get('x7006', '0'))，d.get('x7091', '0')，float(d.get('x006C', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# data_ems = [float(d.get('x7006', '0'))]
		# data_ems = [float(d.get('x7000', '0')),float(d.get('x7006', '0')),float(d.get('x006C', '0')),float(d['x7001'])]
		# data_ems = [float(d.get('x7000', '0')),float(d.get('x7006', '0')),float(d.get('x006C', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# data_label = [float(d.get('x7001', '0'))]
		# try:
		# 	data_label = [float(d['x7002'])]
		# 	tfc = d['x7002']
		# except KeyError:
		# 	data_label = [float(d.get('x7002',tfc))]
		data_ems = [engine,torque]
		data_label = [label]
		print(data_ems)
		X.append(data_ems)
		y.append(data_label)

	return X,y


# define wider model
def cnn_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=2,kernel_initializer='normal', activation='relu'))
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

scaler =  StandardScaler()
# scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)
# print(X)
# print(Y)

X = np.array(X)
Y = np.array(Y)

# n = 5
# poly = PolynomialFeatures(n)# returns: [1, x, x^2, x^3]
# X = poly.fit_transform(X)
# print(X.shape)
# print(X[0])

# checkpoint
filepath="mlp-weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

r = 0

for train_X, train_y, test_X, test_y in kfold_load(X,Y):

	trainX, trainY = [],[]
	# for i in range(len(train_X)-look_back-1):
	# 	a = train_X[i:(i+look_back),:]
	# 	trainX.append(a)
	# 	trainY.append(train_y[i+look_back])

	testX, testY = [],[]
	# for i in range(len(test_X)-look_back-1):
	# 	a = test_X[i:(i+look_back),:]
	# 	testX.append(a)
	# 	testY.append(test_y[i+look_back])
	# print(np.array(trainX).shape)
	# trainX = np.reshape(trainX,(np.array(trainX).shape[0],np.array(trainX).shape[2]*np.array(trainX).shape[1]))
	# testX = np.reshape(testX,(np.array(testX).shape[0],np.array(testX).shape[2]*np.array(testX).shape[1]))

	# print(trainX)
	# print(testX)
	trainX = train_X
	trainY = train_y
	testX = test_X
	testY = test_y

	model = cnn_model()
	# model = cnn_model()
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.1,callbacks=callbacks_list, verbose=1)

	np.savetxt('cnn_loss.csv', history.history['loss'])
	np.savetxt('cnn_val_loss.csv', history.history['val_loss'])

	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	print(trainY)
	print(trainPredict)

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
	model.save('model/cnn/CNN_model_%s+%f.h5'%(t0,testr2))
	# model.summary()
	# plt.plot(testY[0][:1000])
	# plt.plot(testPredict[:1000,0],color="red")
	# plt.show()

print(r/5)
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
	# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01')"
	sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"

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
		data_ems = [float(d.get('x7000', '0')),float(d.get('x7003', '0')),float(d.get('x7004', '0')), 
			float(d.get('x7005', '0')),float(d.get('x7006', '0')),float(d.get('x7007', '0')),float(d.get('x006C', '0')),float(d.get('x7035', '0')),float(d.get('x7091', '0')),float(b['lat'][i]),float(b['lng'][i])]
		# data_label = [float(d.get('x7001', '0'))]
		try:
			data_label = [float(d['x7001'])]
		except KeyError:
			continue
		# try:
		# 	data_label = [float(d['x7002'])]
		# 	tfc = d['x7002']
		# except KeyError:
		# 	continue
			# data_label = [float(d.get('x7002',tfc))]
		# print(data_ems[5])
		X.append(data_ems)
		y.append(data_label)

	return X,y


X,Y = load_emsdata()
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
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the coefficient
of determination are also calculated.

"""
# print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

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
# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (county = '青羊区')"
# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01')"
sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"

#利用pandas 模块导入mysql数据
a=pd.read_sql(sqlcmd,dbconn)
#取前5行数据

X = []
y = []

b = a
tfc = 0
# b = a.head()
for i in range(len(b.index)):
	print(i)
	d = json.loads(b['ems_value'][i])
	try:
		label = float(d['x7001'])
		engine = float(d['x7000'])
		throttle = float(d['x7006'])
		# speed = float(d['x006C'])
	except KeyError:
		continue
	# print(d['x7001'])
	# data_ems = [float(d.get('x7006', '0'))]
	# data_ems = [float(d.get('x7000', '0')),float(d.get('x7006', '0')),float(d.get('x006C', '0'))]
	# data_ems = [float(d.get('x7000', '0')),float(d.get('x7006', '0')),float(d.get('x006C', '0')),float(b['lat'][i]),float(b['lng'][i])]
	# data_ems = [float(d.get('x7000', '0')),float(d.get('x7003', '0')),float(d.get('x7004', '0')), 
	# 	float(d.get('x7005', '0')),float(d.get('x7006', '0')),float(d.get('x7007', '0')),float(d.get('x006C', '0')),float(d.get('x7035', '0')),float(d.get('x7091', '0')),float(b['lat'][i]),float(b['lng'][i])]
	# data_label = [float(d.get('x7001','0'))]
	# try:
	# 	data_label = [float(d['x7002'])]
	# 	tfc = d['x7002']
	# except KeyError:
	# 	data_label = [float(d.get('x7002',tfc))]
	data_ems = [engine,throttle]
	data_label = [label]
	X.append(data_ems)
	y.append(data_label)

def kfold_load(X,y):
	n_splits = 5
	kfold = KFold(n_splits=n_splits, shuffle=False)

	print('KFold = %d :'%n_splits)
	for train_index, test_index in kfold.split(X, y):
	    train_X, test_X = X[train_index], X[test_index]
	    train_y, test_y = y[train_index], y[test_index]
	    yield train_X, train_y, test_X, test_y

y = np.squeeze(y)

X = np.array(X)
Y = np.array(y)

# Split the data into training/testing sets
# X_train = X[:-10000]
# X_test = X[-10000:]
# print(len(X_train))
# print(len(X_test))

# Split the targets into training/testing sets
# y_train = y[:-10000]
# y_test = y[-10000:]
# print(len(y_train))
# print(len(y_test))

r = 0

for train_X, train_y, test_X, test_y in kfold_load(X,y):
	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(train_X, train_y)

	# Make predictions using the testing set
	y_pred = regr.predict(test_X)

	# The coefficients
	print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print('Mean squared error: %.2f' % mean_squared_error(test_y, y_pred))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f' % r2_score(test_y, y_pred))
	r += r2_score(test_y, y_pred)

print(r/5)

# Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
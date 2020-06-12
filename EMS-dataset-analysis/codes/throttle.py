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
from keras.layers import Dense, Reshape, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import math

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
sqlcmd="SELECT ems_value FROM emsdata"
# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (county = '青羊区')"
# sqlcmd="SELECT ems_value FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01')"
# sqlcmd="SELECT ems_value,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '永州市') AND ((tdate = '20191204') or (tdate = '20191209') or (tdate = '20191219') or (tdate = '20191224') or (tdate = '20191228'))"
# sqlcmd="SELECT ems_value FROM emsdata_2 where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (city = '重庆市') AND (tdate = '20200417')"
#利用pandas 模块导入mysql数据
a=pd.read_sql(sqlcmd,dbconn)

b = a
# b = a.head()
tfc = 0
n = 0
for i in range(len(b.index)):
	# print(i)
	d = json.loads(b['ems_value'][i])
	try:
		# data_label = [float(d['x7001'])]
		label = float(d['x7001'])
		engine = float(d['x7000'])
		throttle = float(d['x7006'])
		# torque = float(d['x704F'])
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
	data_ems = [engine,throttle]
	data_label = [label]
	if throttle > 100:
		n += 1
	print(data_ems)
	# X.append(data_ems)
	# y.append(data_label)
print(n)
# a.to_excel("ems-20200417.xlsx")
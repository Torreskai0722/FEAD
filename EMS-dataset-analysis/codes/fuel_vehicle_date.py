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

mile_error = 1
samples = 50

def load_data():
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
	# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') and (city = '永州市')"
	sqlcmd1="SELECT truckid FROM emsdata_2 group by truckid"
	# sqlcmd2="SELECT tdate FROM emsdata_2 group by tdate"
	# sqlcmd2 = "select tdate FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01') and (city = '桂林市') group by tdate"
	#利用pandas 模块导入mysql数据
	s = pd.read_sql(sqlcmd1,dbconn)
	print(s)

	miles = []
	lat = []
	lng = []

	for i in range(len(s['truckid'])):
		sqlcmd = "SELECT truckid,tdate FROM emsdata_2 group by truckid,tdate"
		# sqlcmd = "SELECT city FROM emsdata_2 where truckid = %s group by city" % s['truckid'][i]
		d = []
		# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01') and (city = '桂林市') and (tdate = %s)" % s['tdate'][i]
		a = pd.read_sql(sqlcmd,dbconn)
		for j in range(len(a['truckid'])):
			if s['truckid'][i] == a['truckid'][j]:
				d.append(a['tdate'][j])
		print('-------------------')
		print(s['truckid'][i])
		print(d)
		
load_data()
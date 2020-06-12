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
	# sqlcmd2="SELECT tdate FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') and (city = '永州市') group by tdate"
	sqlcmd2 = "SELECT tdate from emsdata_2 where truckid = '257C0D741E2CDDAFDA1A297FC5AC9964' and city = '重庆市' group by tdate"
	# sqlcmd2 = "SELECT tdate from emsdata_2 where truckid = '107624CE6B76B627818179C74FF206CF' and city = '衡阳市' group by tdate"
	# sqlcmd2 = "SELECT tdate from emsdata_2 where truckid = '107EF28DD023DD7F50AEDA9CA033325B' and city = '荆州市' group by tdate"
	# sqlcmd2 = "SELECT tdate from emsdata_2 where truckid = '107EF28DD023DD7F50AEDA9CA033325B' and city = '长沙市' group by tdate"

	# sqlcmd2 = "select tdate FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01') and (city = '桂林市') group by tdate"

	#利用pandas 模块导入mysql数据
	s = pd.read_sql(sqlcmd2,dbconn)
	print(s)

	miles = []
	lat = []
	lng = []

	for i in range(len(s['tdate'])):
		# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata_2 where (truckid = '107624CE6B76B627818179C74FF206CF') and (city = '衡阳市') and (tdate = %s)" % s['tdate'][i]
		# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata_2 where (truckid = '107EF28DD023DD7F50AEDA9CA033325B') and (city = '荆州市') and (tdate = %s)" % s['tdate'][i]
		# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata_2 where (truckid = '107EF28DD023DD7F50AEDA9CA033325B') and (city = '长沙市') and (tdate = %s)" % s['tdate'][i]
		sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata_2 where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') and (city = '重庆市') and (tdate = %s)" % s['tdate'][i]
		# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') and (city = '永州市') and (tdate = %s)" % s['tdate'][i]
		# sqlcmd="SELECT ems_value,triggertime,lat,lng FROM emsdata where (truckid = 'CABB5C23A2B5E1541DB6E75FD1D61E01') and (city = '桂林市') and (tdate = %s)" % s['tdate'][i]
		a = pd.read_sql(sqlcmd,dbconn)
		print("++++++++++++++++++++++")
		print(s['tdate'][i])
		print(len(a['ems_value']))

		d1 = json.loads(a['ems_value'][0])
		# m1 = float(d1['x7005'])
		m1 = float(d1.get('x7005', '0'))
		print(round(float(a['lat'][0]),3),round(float(a['lng'][0]),3))

		d2 = json.loads(a['ems_value'][len(a['ems_value'])-1])
		# m2 = float(d2['x7005'])
		m2 = float(d2.get('x7005', '0'))
		print(round(float(a['lat'][len(a['lat'])-1]),3),round(float(a['lng'][len(a['lng'])-1]),3))

		# miles = []
		miles.append(round(m2-m1,3))

		le = len(a['lat'])
		delta = int(le / samples)
		tlat = []
		tlng = []
		for j in range(le):
			if j%delta == 0:
				# pst = [round(float(a['lat'][j]),3),round(float(a['lng'][j]),3)]
				tlat.append(round(float(a['lat'][j]),3))
				tlng.append(round(float(a['lng'][j]),3))
		lat.append(tlat)
		lng.append(tlng)
		print(round(m2-m1,3))

	proute_index = []
	routes = []

	# lat = np.array(lat)
	# lng = np.array(lng)

	# print("--------------------------")
	# print(lat)
	# print(lng)

	mm = np.asarray(miles,dtype=np.float32)

	for i in range(len(mm)):
		if miles[i] >= mm.max() - 2 * mile_error:
			proute_index.append(i)
	# print(proute_index)

	for i in range(len(mm)):
		for j in range(i):
			a = np.array(lat[i]) - np.array(lat[j])
			b = np.array(lng[i]) - np.array(lng[j])
			# print("****************************")
			# print(i,j)
			# print(a)
			# print(b)

	# for i in proute_index:
	# 	print(s['tdate'][i])
	# 	print(lat[i])
	# 	print(lng[i])

load_data()
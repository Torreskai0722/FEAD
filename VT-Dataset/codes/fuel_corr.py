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

def data_access(path):

	# file_vehicle_1 = "../veh001_all.csv"
	# file_vehicle_2 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"
	# file_vehicle_3 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"

	df_data = pd.read_csv(path,index_col=False,usecols=[0,1,2,3,4,5,6,7,8],header=0,names=['CO2', 'CO', 'HC', 'NOx', 'vel','fuel', 'engine', 'elevation', 'phase_num'])

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

		data_features = [CO2,CO,HC,NOx,vel,engine,elevation,phase_num]
		# data_features = [CO2,CO,HC,NOx,vel,engine]

		fuel = float(df_data['fuel'][i])
		data_label = [fuel]

		X.append(data_features)
		y.append(data_label)

		# print(i)

	return X,y

paths = ["../veh001_all.csv","../veh002_all.csv","../veh003_all.csv"]

VT_data = []
for path in paths:
    X,Y = data_access(path)
    scaler =  StandardScaler()
	# scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)

    Z = np.hstack((X,Y))

	#C_mat = np.cov(Z)
    C_mat = np.corrcoef(Z.T)
    print(C_mat[8])
    VT_data.append(C_mat[8])

VT_data = np.array(VT_data).transpose()
df_C_mat = pd.DataFrame(data=VT_data,columns=['vehicle-1','vehicle-2','vehicle-3'],index=['CO2','CO','HC','NOx','vel','engine','elevation','phase_num','fuel'])
# print(C_mat.shape)
# print(C_mat)
fig = plt.figure(figsize = (4,10))

sb.heatmap(df_C_mat, vmax = 1, square = True,annot=True,cbar=False)
plt.show()
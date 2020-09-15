#!/usr/bin/python
# -*- coding: utf-8 -*-

# Regression Example With Boston Dataset: Baseline
import pandas as pd
import pymysql
import json
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score
from pandas import read_csv,read_excel
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import math
import time
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.wrappers import TimeDistributed
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import autosklearn.classification


look_back = 4

def data_access():

	file_vehicle_1 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"
	# file_vehicle_2 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"
	# file_vehicle_3 = "/home/edge22/projects/fuel efficiency/VT-dataset/veh001_all.csv"

	df_data = pd.read_csv(file_vehicle_1,index_col=False,usecols=[0,1,2,3,4,5,6,7,8],header=0,names=['CO2', 'CO', 'HC', 'NOx', 'vel','fuel', 'engine', 'elevation', 'phase_num'])

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

		# data_features = [CO2,CO,HC,NOx,vel,engine,elevation,phase_num]
		data_features = [CO2,CO,HC,NOx,vel,engine]

		fuel = float(df_data['fuel'][i])
		data_label = [fuel]

		X.append(data_features)
		y.append(data_label)

		# print(i)

	return X,y

X, y = data_access()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)

print("R2 score", r2_score(y_test, y_hat))
print("Accuracy score", accuracy_score(y_test, y_hat))
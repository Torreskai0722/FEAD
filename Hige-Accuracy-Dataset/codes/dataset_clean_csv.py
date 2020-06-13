#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import csv
import os
import pandas as pd


def data_access():
	# file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
	# file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"

	file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
	file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"
	file_fuel_rate = "/Users/torres_kai/Downloads/fuel-dataset-3/0x721_Ins_flow_rate.csv"

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	# df_location = pd.read_csv(file_localization,index_col=False,header=0)
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0],names=['time'])
	# df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	')
	df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])

	print(df_rate)
	print(len(df_vehicle['time']))
	t0 = 1590556664.10647
	j = 1
	n = 0

	for i in range(len(df_rate['time'])):
		t = t0 + df_rate['time'][i]
		if j >= len(df_vehicle['time']) - 1:
    			break
		while j < len(df_vehicle['time']) and float(df_vehicle['time'][j])/1000000000 < t:
    			j = j + 1
		print(j)
		print(n)
		print(t, float(df_vehicle['time'][j])/1000000000, t - float(df_vehicle['time'][j])/1000000000)
		n = n + 1

	# for i in range(len(df_vehicle['publish time'])):
	#     	print(df_vehicle['publish time'][i])

data_access()
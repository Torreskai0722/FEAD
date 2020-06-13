#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import csv
import os
import pandas as pd


def data_access():
	file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
	file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"
	file_fuel_rate = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/0x721_Ins_flow_rate.csv"

	# file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
	# file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"
	# file_fuel_rate = "/Users/torres_kai/Downloads/fuel-dataset-3/0x721_Ins_flow_rate.csv"

	df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
	# df_location = pd.read_csv(file_localization,index_col=False,header=0)
	df_vehicle = pd.read_csv(file_vehicle_report,index_col=False,usecols=[0,23,24,25,26,27,33,37],names=['time','throttle_position_percent',
		'engine_torque_percent','driver_demand_engine_torque_percent','engine_torque_loss_percent','engine_speed_rpm','combined_vehicle_weight_kg', 'vehicle_speed_mps'])
	# df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	')
	df_rate = pd.read_csv(file_fuel_rate,index_col=False,header=None,sep='	',usecols=[0,1],names = ['time','fuel_rate'])

	print(df_rate)
	print(len(df_vehicle['time']))
	t0 = 1590556664.10647
	j = 1
	n = 0

	X = []
	y = []

	for i in range(len(df_rate['time'])):
		t = t0 + df_rate['time'][i]
		if j >= len(df_vehicle['time']) - 1:
    			break
		while j < len(df_vehicle['time']) and float(df_vehicle['time'][j])/1000000000 < t:
    			j = j + 1
		print(j)
		print(n)

		throttle_position_percent = df_vehicle['throttle_position_percent'][j-1]
		engine_torque_percent = df_vehicle['engine_torque_percent'][j-1]
		driver_demand_engine_torque_percent = df_vehicle['driver_demand_engine_torque_percent'][j-1]
		engine_torque_loss_percent = df_vehicle['engine_torque_loss_percent'][j-1]
		engine_speed_rpm = df_vehicle['engine_speed_rpm'][j-1]
		combined_vehicle_weight_kg = df_vehicle['combined_vehicle_weight_kg'][j-1]
		vehicle_speed_mps = df_vehicle['vehicle_speed_mps'][j-1]

		fuel_rate = df_rate['fuel_rate'][i]

		data_ems = [throttle_position_percent,engine_torque_percent,driver_demand_engine_torque_percent,engine_torque_loss_percent,engine_speed_rpm, combined_vehicle_weight_kg, vehicle_speed_mps]
		data_label = [fuel_rate]

		X.append(data_ems)
		y.append(data_label)

		# print(t, float(df_vehicle['time'][j-1])/1000000000, t - float(df_vehicle['time'][j-1])/1000000000)

		n = n + 1

	return X,y

	# for i in range(len(df_vehicle['publish time'])):
	#     	print(df_vehicle['publish time'][i])

X,y = data_access()
print(X)
print(y)
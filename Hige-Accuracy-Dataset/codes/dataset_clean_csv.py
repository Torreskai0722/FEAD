#!/usr/bin/env python
#-*- coding: utf-8 -*-
import csv
import os
import pandas as pd

# file_localization = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/localization_result.csv"
# file_vehicle_report = "/home/edge22/projects/fuel efficiency/Fuel-dataset-3/vehicle_report.csv"

file_localization = "/Users/torres_kai/Downloads/fuel-dataset-3/localization_result.csv"
file_vehicle_report = "/Users/torres_kai/Downloads/fuel-dataset-3/vehicle_report.csv"

df_location = pd.read_csv(file_localization,index_col=False,usecols=[0,1,2,3],header=0,names = ['F','S','T','G'])
# df_location = pd.read_csv(file_localization,index_col=False,header=0)
df_vehicle = pd.read_csv(file_vehicle_report,index_col=False)

print(df_location)
print(len(df_location['F']))

# for i in range(len(df_location['F'])):
for i in range(0,100):
	print(df_location['F'][i]/1000000,df_location['S'][i]/1000000,(df_location['S'][i] - df_location['F'][i])/1000000)

# for j in range(len(df_location['receive_ts'])):
# 	print(df_location['receive_ts'][j])
	# print(df_location['receive_ts'][i])
	# df_location['unix_pub_ts'][i],df_location['gnss_m_ts_ns'][i])
	# i = i + 1
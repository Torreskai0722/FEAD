# -*- coding: utf-8 -*-

import pandas as pd
import pymysql
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl
import csv

def write_csv():
    path  = "aa.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = ["1","2"]
        csv_write.writerow(data_row)
 
 
book_name_vvid = '257C0D741E2CDDAFDA1A297FC5AC9964.xlsx'
 
sheet_name_date = '20191201'
sheet_name_label = '20191201-label' 
 
# write_excel_xlsx(book_name_vvid, sheet_name_date, datavalue)
# read_excel_xlsx(book_name_vvid, sheet_name_date)

## 加上字符集参数，防止中文乱码
dbconn=pymysql.connect(
  host="localhost",
  database="fuel_efficiency",
  user="root",
  password="123456",
  port=3306,
  charset='utf8'
 )

#sql语句
sqlcmd="SELECT ems_value FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (tdate = '20191201')"

#利用pandas 模块导入mysql数据
a=pd.read_sql(sqlcmd,dbconn)
#取前5行数据

X = []
y = []

# b = a
b = a.head()
for i in range(len(b.index)):
	print(i)
	d = json.loads(b['ems_value'][i])
	data_ems = [float(d.get('x7000', '0')),float(d.get('x7002', '0')),float(d.get('x7003', '0')),float(d.get('x7004', '0')), 
		float(d.get('x7005', '0')),float(d.get('x7006', '0')),float(d.get('x7007', '0')),float(d.get('x006c', '0')),float(d.get('x7035', '0')),float(d.get('x7091', '0')),float(b['lat'][i]),float(b['lng'][i])]
	data_label = [float(d.get('x7001', '0'))]
	print(type(d.get('x7001', '0')))
	X.append(data_ems)
	y.append(data_label)
	# write_excel_xlsx(book_name_vvid, sheet_name_date, data_ems)
	# write_excel_xlsx(book_name_vvid, sheet_name_label, data_label)
	# print(d.get('x7000', '0'),d.get('x7001', '0'),d.get('x7002', '0'),d.get('x7003', '0'),d.get('x7004', '0'), 
	# 	d.get('x7005', '0'),d.get('x7006', '0'),d.get('x7007', '0'),d.get('x006c', '0'),d.get('x7035', '0'),d.get('x7091', '0'))
	# print(b['lat'][i])
	# print(b['lng'][i])

# xls = pd.ExcelFile(book_name_vvid)
# df1 = pd.read_excel(xls, sheet_name_date)
# X = np.array(df1)
# df2 = pd.read_excel(xls, sheet_name_label)
# y = np.array(df2)
print(len(b.index))
print(X)
print(y)
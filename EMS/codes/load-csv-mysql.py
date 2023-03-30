#!/usr/bin/env python
#-*- coding: utf-8 -*-
import csv
import os
import pandas as pd
from pandas.io import sql
# from sqlalchemy import create_engine

with open('dwa_inceptio_ems_detail_location_manual.csv') as fin:    
    csvin = csv.DictReader(fin)
    # Category -> open file lookup
    outputs = {}
    for row in csvin:
    	print(row)
        cat = row
        # Open a new file and write the header
        if cat not in outputs:
            fout = open('{}.csv'.format(cat), 'w')
            dw = csv.DictWriter(fout, fieldnames=csvin.fieldnames)
            dw.writeheader()
            outputs[cat] = fout, dw
        # Always write the row
        outputs[cat][1].writerow(row)
    # Close all the files
    for fout, _ in outputs.values():
        fout.close()

# i = 0
# p = []
# T = []
# d = {}
# length = 0
# for info in os.listdir('/Users/torres_kai/Downloads/fuel_efficiency/split/'):
# 	if 'csv' in info:
# 	    domain = os.path.abspath(r'/Users/torres_kai/Downloads/fuel_efficiency/split/') #获取文件夹的路径
# 	    info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
# 	    # print(info)
# 	    dfcolumns = pd.read_csv(info, nrows = 1)
# 	    # print(dfcolumns)
# 	    # data = pd.read_csv(info, encoding='utf-8', delim_whitespace=True, header = None, skiprows = 1, usecols = list(range(len(dfcolumns.columns))))
# 	    data = pd.read_csv(info, encoding='utf-8', sep='	', header = None, usecols=[8,10,11,13,20], names = ['P','Lat','Lng','T','D'])
# 	    # print(data)
# 	    # dd = data.to_string()
# 	    # for j in range(len(data['D'])):
# 	    # 	T.append(data['D'][j])
# 	    length += len(data['P'])

# 	    for j in range(len(data['P'])):
# 	    	p.append([data['P'][j],data['D'][j]])

# 	    # for j in range(len(data['P'])):
# 	    # 	p.append([data['P'][j],[data['Lat'][j],data['Lng'][j]]])
# 	    # for j in data['P']:
# 	    # 	p.append(j)
# 	    # print(data)
# 	    i += 1
# 	    print(i)

# print(length)
# total: 10273626 rows
# df = pd.read_csv('File.csv', usercols=['ID', 'START_DATE'], skiprows=skip)
# print(df)

# engine = create_engine('mysql://username:password@localhost/dbname')
# with engine.connect() as conn, conn.begin():
#     df.to_sql('Table1', conn, if_exists='replace')


# delim_whitespace=True
# dfcolumns = pd.read_csv('file.csv', nrows = 1)
# df = pd.read_csv('file.csv',
#                   header = None,
#                   skiprows = 1,
#                   usecols = list(range(len(dfcolumns.columns))),
#                   names = dfcolumns.columns)
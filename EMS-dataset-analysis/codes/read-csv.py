#!/usr/bin/env python
#-*- coding: utf-8 -*-
import csv
import os
import pandas as pd
# from path import path

# with open("csv.csv",'r',encoding="utf-8") as f:
#     reader = csv.reader(f)
#     fieldnames = next(reader)#获取数据的第一列，作为后续要转为字典的键名 生成器，next方法获取
#     # print(fieldnames)
#     csv_reader = csv.DictReader(f,fieldnames=fieldnames) #self._fieldnames = fieldnames   # list of keys for the dict 以list的形式存放键名
#     for row in csv_reader:
#         d={}
#         for k,v in row.items():
#             d[k]=v
#         print(d)
i = 0
p = []
T1 = []
T2 = []
T3 = []
T4 = []
T5 = []
T6 = []
T7 = []
T8 = []
T9 = []
TR = []
vid1 = []
vid2 = []
vid3 = []
vid4 = []
vid5 = []
vid6 = []
vid7 = []
vid8 = []
vid9 = []
TR = []
d = {}
length = 0
T = []
for info in os.listdir('/Users/torres_kai/Downloads/fuel_efficiency/split/'):
	if 'csv' in info:
	    domain = os.path.abspath(r'/Users/torres_kai/Downloads/fuel_efficiency/split/') #获取文件夹的路径
	    info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
	    # print(info)
	    dfcolumns = pd.read_csv(info, nrows = 1)
	    # print(dfcolumns)
	    # data = pd.read_csv(info, encoding='utf-8', delim_whitespace=True, header = None, skiprows = 1, usecols = list(range(len(dfcolumns.columns))))
	    data = pd.read_csv(info, encoding='utf-8', sep='	', header = None, usecols=[5,8,13,20], skiprows = 1, names = ['vid','P','T','D'])
	    # data = pd.read_csv(info, encoding='utf-8', sep='	', header = None, usecols=[5,8], names = ['vid','P'])
	    # for j in range(len(data['P'])):
	    # 	# print(data['P'][j],data['vid'][j])
	    # 	T.append(data['vid'][j])
	    # dd = data.to_string()
	    for j in range(len(data['D'])):
	    	if data['P'][j] == u'粤ABP691':
	    		T1.append(data['D'][j])
	    		vid1.append(data['vid'][j])
	    	elif data['P'][j] == u'闽K59769':
	    		T2.append(data['D'][j])
	    		vid2.append(data['vid'][j])
	    	elif data['P'][j] == u'粤ADS670':
	    		T3.append(data['D'][j])
	    		vid3.append(data['vid'][j])
	    	elif data['P'][j] == u'粤ADW293':
	    		T4.append(data['D'][j])
	    		vid4.append(data['vid'][j])
	    	elif data['P'][j] == u'闽K59938':
	    		T5.append(data['D'][j])
	    		vid5.append(data['vid'][j])
	    	elif data['P'][j] == u'闽K59936':
	    		T6.append(data['D'][j])
	    		vid6.append(data['vid'][j])
	    	elif data['P'][j] == u'粤ADP980':
	    		T7.append(data['D'][j])
	    		vid7.append(data['vid'][j])
	    	elif data['P'][j] == u'粤ABW222':
	    		T8.append(data['D'][j])
	    		vid8.append(data['vid'][j])
	    	elif data['P'][j] == u'闽K55572':
	    		T9.append(data['D'][j])
	    		vid9.append(data['vid'][j])
	    	else:
	    		TR.append(data['D'][j])

	    length += len(data['P'])
	    # for j in range(len(data['P'])):
	    # 	p.append([data['P'][j],data['D'][j]])

	    # for j in range(len(data['P'])):
	    # 	p.append([data['P'][j],[data['Lat'][j],data['Lng'][j]]])
	    # for j in data['P']:
	    # 	p.append(j)
	    # print(data)
	    i += 1
	    print(i)

# print(T)
# print(list(set(T)))

# print(p)
# print(list(set(p)))
# ['粤ABP691', '闽K59769', '粤ADS670', '粤ADW293', '闽K59938', '闽K59936', '粤ADP980', '粤ABW222', '闽K55572'] 9 trucks

# [u'3CCC005122531737E69EA3BA21324ECD', 
# u'A0A4A31F4C3509655240D8D7DB9CD389', 
# u'83E94E04A08895767DFE0D80A21A07D3', 
# u'B45F36E3944670E22C7EC735D833F709', 
# u'7571522FF0EBA036818CEACBA52D3B60', 
# u'096B3BBA5216C10C7EDF72FD803ACFD7', 
# u'3EF3F915B5831AE8667B6FC54FBA89B7', 
# u'CABB5C23A2B5E1541DB6E75FD1D61E01', 
# u'257C0D741E2CDDAFDA1A297FC5AC9964']

print(TR)
print("粤ABP691:")
print(list(set(T1)))
print(list(set(vid1)))
print("闽K59769:")
print(list(set(T2)))
print(list(set(vid2)))
print("粤ADS670:")
print(list(set(T3)))
print(list(set(vid3)))
print("粤ADW293:")
print(list(set(T4)))
print(list(set(vid4)))
print("闽K59938:")
print(list(set(T5)))
print(list(set(vid5)))
print("闽K59936:")
print(list(set(T6)))
print(list(set(vid6)))
print("粤ADP980")
print(list(set(T7)))
print(list(set(vid7)))
print("粤ABW222")
print(list(set(T8)))
print(list(set(vid8)))
print("闽K55572:")
print(list(set(T9)))
print(list(set(vid9)))

# 粤ABP691:
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# 闽K59769:
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# 粤ADS670:
# [20191228, 20191229, 20191230, 20191231]
# 粤ADW293:
# [20191227, 20191228, 20191229, 20191230, 20191231]
# 闽K59938:
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# 闽K59936:
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# 粤ADP980
# [20191203, 20191204, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# 粤ABW222
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# 闽K55572:
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]
# [20191201, 20191202, 20191203, 20191204, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231]

# 30 days missing 20191205

# Total: 10,273,969 rows

print(length)


# delim_whitespace=True
# dfcolumns = pd.read_csv('file.csv', nrows = 1)
# df = pd.read_csv('file.csv',
#                   header = None,
#                   skiprows = 1,
#                   usecols = list(range(len(dfcolumns.columns))),
#                   names = dfcolumns.columns)
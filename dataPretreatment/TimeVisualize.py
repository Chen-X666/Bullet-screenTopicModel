# _*_ coding: utf-8 _*_
"""
Time:     2022/1/10 17:52
Author:   ChenXin
Version:  V 0.1
File:     TimeVisualize.py
Describe:  Github link: https://github.com/Chen-X666
"""
from collections import Counter
from sklearn.preprocessing import scale
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def basicMsg():
    df = pd.read_csv('',encoding='utf-8')
    print('弹幕数量')
    print('弹幕用户数')
    print('人均发送弹幕量')
    print('色彩种类')
    print('字幕弹幕数量')

def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total)/lenth
    tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
    tempsum = pow(float(tempsum)/lenth,0.5)
    for i in range(lenth):
        data[i] = (data[i] - ave)/tempsum
    return data

def processTime(df,column):
    #pandas 返回的是格林威治标准时间，与北京时间差了 8 小时
    df[column]=pd.to_datetime(df[column],unit='s', origin='1970-01-01 08:00:00')
    return df

def innerTimeVisualize(input):
    #input = input[input['bvno'] == 'BV1GW411g7mc']
    #print(input)
    # 降序排列
    input = input.sort_values(by=['dm_time'])
    # 按秒分
    input["dm_time"] = input["dm_time"].apply(
        lambda x: str(x).split(".")[0])
    videoTime_data = input["dm_time"].drop_duplicates()
    videoTime_group = input.groupby(["dm_time"])
    x_axis = []
    y_axis = []
    for each_videoTime in videoTime_data:
        data_each_day = videoTime_group.get_group(each_videoTime)
        x_axis.append(int(float(each_videoTime)))
        y_axis.append(len(data_each_day))
    sns.set()
    sns.lineplot(x=x_axis, y=y_axis)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.xticks(np.arange(min(x_axis), max(x_axis), (max(x_axis) - min(x_axis)) / 10))

    plt.xlabel("视频时间(单位：秒)")

    plt.ylabel("弹幕数量")
    plt.show()



def outterTimeVisualize(inputs):
    print(inputs.dtypes)
    #inputs['sendTime'] = inputs['sendTime'].astype('int64')
    print(len(inputs))
    # 数据正排序
    inputs = inputs.sort_values(by=['sendTime'])  # 降序排列
    # 时间搓转换
    #inputs = processTime(inputs, 'sendTime')
    # 把time列的日期时间根据 空格符
    inputs["sendTime"] = inputs["sendTime"].apply(
        lambda x: str(x).split(" ")[1]).apply(
        lambda x: str(x).split(":")[0])
    # 获得不重复的天
    # day_data = inputs["dm_sendTime"].drop_duplicates()
    # day_group = inputs.groupby(["dm_sendTime"])

    # for each_day in day_data:
    #     data_each_day = day_group.get_group(each_day)
    #     x_axis.append(each_day)
    #     y_axis.append(len(data_each_day))
    day_data = inputs["sendTime"].to_list()
    data = pd.DataFrame.from_dict(Counter(day_data), orient='index').reset_index().rename(
        columns={'index': '弹幕日期', 0: '弹幕数量'})
    #dataset.to_csv('dataset/sendTimeData.csv',encoding='utf-8')
    sns.set()
    sns.lineplot(x="弹幕日期", y="弹幕数量",  data=data).figure.set_size_inches(16, 6)
    plt.gcf().autofmt_xdate()

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # plt.xticks(['2019-12-16','2018-12-17',
    #             '2019-12-18','2019-05-01',
    #             '2020-01-01','2020-06-01',
    #             '2020-12-01'])

    #plt.xticks(np.arange(min(x_axis),max(x_axis),(max(x_axis)-min(x_axis))/5))
    # get current axis


    plt.show()

def countTimeData(filePath,timeColumn,startTime,endTime,numColumn):
    df = pd.read_csv(filePath,encoding='utf-8')
    df = df[(df[timeColumn]>=startTime)&(df[timeColumn]<=endTime)]
    print(sum(df[numColumn].to_list()))
    return sum(df[numColumn].to_list())




if __name__ == '__main__':
    df = pd.read_csv('dataset/yq2019-12-01_2020-12-01.csv', encoding='utf-8')
    df['sendTime'] = pd.to_datetime(df['sendTime'], unit='s', origin='1970-01-01 08:00:00')
    newDf = df[df['sendTime'] >= '2019-12-17 00:00:00']
    newDf = newDf[newDf['sendTime'] <= '2019-12-17 23:59:59']
    outterTimeVisualize(inputs=newDf)
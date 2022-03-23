# _*_ coding: utf-8 _*_
"""
Time:     2022/3/1 15:46
Author:   ChenXin
Version:  V 0.1
File:     divideDataVisualize.py
Describe:  Github link: https://github.com/Chen-X666
"""
from collections import Counter
from sklearn.preprocessing import scale
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def processTime(df,column):
    #pandas 返回的是格林威治标准时间，与北京时间差了 8 小时
    df[column]=pd.to_datetime(df[column],unit='s', origin='1970-01-01 08:00:00')
    return df

#潜伏期 2019.12.01 00:00:00 —— 2020.01.20 23:59:59 1575129600——1579535999
#爆发期1 2020.01.20 00:00:00—2020.03.13 23:59 1579449600——1584115199
#爆发期2 2020.03.12—2020.03.13 1577815200——1577818800
#缓和期 2020.03.12—2020.05.06 1577818800——1577826000

def divideVisualize(inputs,startData,endData):
    # print(inputs.dtypes)
    # #inputs['sendTime'] = inputs['sendTime'].astype('int64')
    # # inputs = inputs[inputs['sendTime'] >= 1575129600]
    # # inputs = inputs[inputs['sendTime'] <= 1577826000]
    # print(len(inputs))
    # # 数据正排序
    # inputs = inputs.sort_values(by=['sendTime'])  # 降序排列
    # # 时间搓转换
    # inputs = processTime(inputs, 'sendTime')
    # # 把time列的日期时间根据 空格符
    # inputs["sendTime"] = inputs["sendTime"].apply(
    #     lambda x: str(x).split(" ")[0])
    # inputs = inputs[inputs['sendTime'] >= '2019-12-01 00:00:00']
    # inputs = inputs[inputs['sendTime'] <= '2020-12-01 23:59:59']
    # inputs['sendTime'] = pd.to_datetime(inputs['sendTime'])
    # day_data = inputs["sendTime"].to_list()
    # dataset = pd.DataFrame.from_dict(Counter(day_data), orient='index').reset_index().rename(
    #     columns={'index': '弹幕日期', 0: '弹幕数量'}).set_index('弹幕日期')
    # dateRange = pd.date_range(start=startData, end=endData)
    # print(dateRange)
    # dataset = dataset.reindex(dateRange)
    # print(dataset)
    # dataset['弹幕数量'] = dataset['弹幕数量'].fillna(0)
    # print(dataset)
    # dataset.to_csv('dataset/sendTimeData.csv', encoding='utf-8')
    data = pd.read_csv('dataset/sendTimeData.csv', encoding='utf-8')
    print(sum(data['弹幕数量'].to_list()))
    # dataset = dataset[dataset['弹幕日期'] >= '2019-12-01']
    # dataset = dataset[dataset['弹幕日期'] <= '2020-12-01']
    sns.set()
    sns.lineplot(x='弹幕日期', y='弹幕数量', data=data).figure.set_size_inches(12, 6)
    plt.gcf().autofmt_xdate()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # plt.xticks(['2020-01-20',
    #             '2019-12-16', '2019-12-17',
    #             '2019-12-18', '2019-12-19',
    #             '2019-12-23',
    #             '2020-03-12'])
    # plt.xticks(['2019-12-01',
    #             '2020-01-21',
    #             '2020-03-14',
    #             '2020-04-07',
    #             '2020-08-15',
    #             '2020-10-01',
    #             '2020-12-01'])
    plt.xticks(['2019-12-01',
                '2019-12-10',
                '2019-12-20',
              #  '2020-04-07',
                '2020-01-01',
                '2020-01-10',
                '2020-01-20'])
    #plt.xticks(np.arange(min(x_axis),max(x_axis),(max(x_axis)-min(x_axis))/5))
    #plt.xlabel('日期时间')
    plt.show()

if __name__ == '__main__':
    inputs = pd.read_csv('dataset/yq2019-12-01_2020-12-01.csv', encoding='utf-8')
    divideVisualize(inputs=inputs,startData='2019-12-01',endData='2020-01-20')
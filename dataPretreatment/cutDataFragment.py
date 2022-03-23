# _*_ coding: utf-8 _*_
"""
Time:     2022/1/8 10:14
Author:   ChenXin
Version:  V 0.1
File:     cutDataFragment.py
Describe:  Github link: https://github.com/Chen-X666
"""
import jieba
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataPretreatment.preprocessText import seg_depart


def processTime(df,column):
    #pandas 返回的是格林威治标准时间，与北京时间差了 8 小时
    df[column]=pd.to_datetime(df[column],unit='s', origin='1970-01-01 08:00:00')
    return df

def cutDataByMonthByDay(bvno):
    data = []
    inputs = pd.read_csv('dataset/' + bvno, encoding='utf-8')
    # 数据正排序
    inputs = inputs.sort_values(by=['dm_sendTime'])  # 降序排列
    # 时间搓转换
    inputs = processTime(inputs, 'dm_sendTime')
    # 把time列的日期时间根据 空格符
    inputs["dm_sendTime"] = inputs["dm_sendTime"].apply(
        lambda x: str(x).split(" ")[0])
    #进行分列获得日期，并形成原始数据新的date一列
    inputs["dm_sendData"] = inputs["dm_sendTime"].astype('datetime64[M]')
    #获得不重复的月与天
    month_data = inputs["dm_sendData"].drop_duplicates()
    # 根据date列进行分组，这个结果你是看不到的
    day_group = inputs.groupby(["dm_sendTime"])
    print('一共有{m}个月'.format(m=len(month_data)))
    # 根据不重复的日期循环，获得相应日期的所有数据
    for each_month in month_data:
        print(each_month)
        data = inputs [inputs["dm_sendData"] == each_month]
        # 写入文件
        outputs = open('dataset/cutPeriodData/' + bvno + '-' + str(each_month).split(' ')[0] + '.txt', 'w', encoding='UTF-8')
        day_data = data["dm_sendTime"].drop_duplicates()
        print('{m}月有{d}个天'.format(m=each_month, d=len(day_data)))
        for each_day in day_data:
            data_each_day = day_group.get_group(each_day)
            lines = data_each_day['dm_text'].to_list()
            for line in lines:
                line_seg = seg_depart(line)
                if line_seg != '':
                    outputs.write(line_seg)
            outputs.write('\n')
        outputs.close()

def cutDataByDay(bvno):
    data = []
    inputs = pd.read_csv('dataset/' + bvno, encoding='utf-8')
    # 数据正排序
    inputs = inputs.sort_values(by=['dm_sendTime'])  # 降序排列
    # 时间搓转换
    inputs = processTime(inputs, 'dm_sendTime')
    # 把time列的日期时间根据 空格符
    inputs["dm_sendTime"] = inputs["dm_sendTime"].apply(
        lambda x: str(x).split(" ")[0])
    # 获得不重复的月与天
    day_data = inputs["dm_sendTime"].drop_duplicates()
    # 根据date列进行分组，这个结果你是看不到的
    day_group = inputs.groupby(["dm_sendTime"])
    # 根据不重复的日期循环，获得相应日期的所有数据
    for each_day in day_data:
        print('{m}'.format(m=each_day))
        # 写入文件
        outputs = open('dataset/cutPeriodData/' + bvno + '-' + str(each_day).split(' ')[0] + '.txt', 'w',
                       encoding='UTF-8')
        data_each_day = day_group.get_group(each_day)
        lines = data_each_day['dm_text'].to_list()
        for line in lines:
            line_seg = seg_depart(line)
            if line_seg != '':
                outputs.write(line_seg)
                outputs.write('\n')
        outputs.close()

def cutDataByBvno(bvno):
    lines = pd.read_csv('dataset/' + bvno,encoding='utf-8')['dm_text'].to_list()
    outputs = open('dataset/cutPeriodData/' + bvno + '.txt', 'w',
                   encoding='UTF-8')
    for line in lines:
        line_seg = seg_depart(line)
        if line_seg != '':
            outputs.write(line_seg)
            outputs.write('\n')
    outputs.close()

def z_score(x):
    return (x - np.mean(x)) / np.std(x, ddof=1)

def cutVideoTime(inputs=None):
    inputs = pd.read_csv('../dataPretreatment/dataset/疫情潜伏期.csv', encoding='utf-8')
    bvno_group = inputs.groupby(["bvno"])
    bvno_data = inputs["bvno"].drop_duplicates()
    data = pd.DataFrame(
        columns=['bvno', 'sendUserCrc32id', 'sendTime', 'text', 'sendVideoTime', 'color', 'fontSize', 'weight', 'id',
                 'idStr', 'action', 'mode', 'isSub', 'pool', 'attr'])
    for each_bvno in bvno_data:
        print(each_bvno)
        data_each_bvno = bvno_group.get_group(each_bvno)
        inputs = data_each_bvno.sort_values(by=['sendVideoTime'])  # 降序排列
        inputs["sendVideoTime"] = inputs["sendVideoTime"].apply(
            lambda x: str(x).split(".")[0])
        videoTime_data = inputs["sendVideoTime"].drop_duplicates()
        videoTime_group = inputs.groupby(["sendVideoTime"])
        x_axis = []
        y_axis = []
        for each_videoTime in videoTime_data:
            data_each_day = videoTime_group.get_group(each_videoTime)
            x_axis.append(int(float(each_videoTime)))
            y_axis.append(len(data_each_day))
        y_axis = z_score(y_axis)
        sns.set()
        sns.lineplot(x=x_axis, y=y_axis)
        plt.xticks(np.arange(min(x_axis), max(x_axis), (max(x_axis) - min(x_axis)) / 10))
        plt.show()
        xyDict = dict(zip(x_axis, y_axis))
        keys = dict((k, v) for k, v in xyDict.items() if v >= 0).keys()
        inputs['sendVideoTime'] = inputs['sendVideoTime'].astype("int")
        print(inputs)
        print(keys)
        for key in keys:
            #print(inputs[(inputs['sendVideoTime'] >= key)&(inputs['sendVideoTime'] <= key)])
            data = pd.concat([data,inputs[(inputs['sendVideoTime'] >= key)&(inputs['sendVideoTime'] <= key)]],ignore_index=True)
    data.to_csv('test.csv',index=False)



if __name__ == '__main__':
    #inputFilePath = "BV1m7411F7si.csv"
    cutVideoTime(1)


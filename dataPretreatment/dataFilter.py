# _*_ coding: utf-8 _*_
import time

import pandas as pd
import jieba
from dataPretreatment.preprocessText import *

def processTime(df,column):
    #pandas 返回的是格林威治标准时间，与北京时间差了 8 小时
    df[column]=pd.to_datetime(df[column],unit='s', origin='1970-01-01 08:00:00')
    return df

#返回不同数据切片的数组格式
def getDataByLine(inputFilePath, outputFilePath):
    inputs = pd.read_csv(inputFilePath, encoding='utf-8')['dm_text'].to_list()
    #数组按行写入
    outputs = open(outputFilePath, 'w', encoding='UTF-8')
    # 将输出结果写入ou.txt中
    for line in inputs:
        line_seg = seg_depart(line)
        if line_seg!='':
            outputs.write(line_seg + '\n')
    outputs.close()

#不同用户不同文档
def getDataByUser(inputFilePath,outputFilePath):
    data = []
    inputs = pd.read_csv(inputFilePath, encoding='utf-8')
    #获取不同用户
    user_data = inputs["dm_userId"].drop_duplicates()
    # 根据date列进行分组，这个结果你是看不到的
    user_group = inputs.groupby(["dm_userId"])
    print('一共有{d}个用户'.format(d=len(user_group)))
    for each_user in user_data:
        user_each_date = user_group.get_group(each_user)
        data.append(user_each_date['dm_text'].to_list())
    # 写入文件
    outputs = open(outputFilePath, 'w', encoding='UTF-8')
    # 遍历每一个日期
    for i in data:
        # 遍历每个日期下的弹幕
        for line in i:
            line_seg = seg_depart(line)
            if line_seg != '':
                outputs.write(line_seg)
        outputs.write('\n')
    outputs.close()

def getDataByVideotime(filename):
    inputs = pd.read_csv(filename, encoding='utf-8')
    #pd.date_range()
    inputs = processTime(inputs,'dm_sendTime')
    print(inputs)

def getDataByDatatime(inputFilePath, outputFilePath):
    data = []
    inputs = pd.read_csv(inputFilePath, encoding='utf-8')
    #数据正排序
    inputs = inputs.sort_values(by=['dm_sendTime'])  # 降序排列
    #时间搓转换
    inputs = processTime(inputs, 'dm_sendTime')
    print(inputs)
    inputs["dm_sendTime"] = inputs["dm_sendTime"].apply(lambda x:str(x).split(" ")[0])  # 把time列的日期时间根据 空格符 进行分列获得日期，并形成原始数据新的date一列
    # 获得不重复的时期
    date_data = inputs["dm_sendTime"].drop_duplicates()
    # 根据date列进行分组，这个结果你是看不到的
    data_group = inputs.groupby(["dm_sendTime"])
    print('一共有{d}个日期'.format(d = len(date_data)))
    # 根据不重复的日期循环，获得相应日期的所有数据
    for each_date in date_data:
        data_each_date = data_group.get_group(each_date)
        data.append(data_each_date['dm_text'].to_list())
    #写入文件
    outputs = open(outputFilePath, 'w', encoding='UTF-8')
    #遍历每一个日期
    for i in data:
        #遍历每个日期下的弹幕
        for line in i:
            line_seg = seg_depart(line)
            if line_seg != '':
                outputs.write(line_seg)
        outputs.write('\n')
    outputs.close()

if __name__ == '__main__':
    # 给出文档路径
    inputFilePath = "dataset/BV1St4y1y78p.csv"
    outputFilePath = "distributionByUser.txt"
    getDataByUser(inputFilePath,outputFilePath)

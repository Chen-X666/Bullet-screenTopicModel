# _*_ coding: utf-8 _*_
"""
Time:     2022/2/2 16:09
Author:   ChenXin
Version:  V 0.1
File:     main.py
Describe:  Github link: https://github.com/Chen-X666
"""
import random

from dataPretreatment.preprocessText import preprocess_text
import pandas as pd

from dataPretreatment.sen2vec import dimensionReduction
from dataPretreatment.sen2vec.DBSCANCluster import DBSCANCluster, DBSCANCluster_plt
from dataPretreatment.sen2vec.sen2TFIDF import tfIDF

if __name__ == '__main__':
    data = pd.read_csv('../dataset/疫情潜伏期.csv')
    sentences = []
    preprocess_text(data['text'].to_list(), sentences)
    random.shuffle(sentences)
    for sentence in sentences[:10]:
        print(sentence)
    weight = tfIDF(sentences)
    print(len(weight))
    print(weight)
    pca_data = dimensionReduction.pca(weight=weight,delimension=3763)
    DBSCANCluster_plt(pca_data)
    result = DBSCANCluster(pca_data)
    data['class'] = result
    data.to_csv('../proData/疫情潜伏期.csv', index=False)

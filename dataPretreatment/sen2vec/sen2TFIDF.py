# _*_ coding: utf-8 _*_
"""
Time:     2022/2/2 13:47
Author:   ChenXin
Version:  V 0.1
File:     sen2TFIDF.py
Describe:  Github link: https://github.com/Chen-X666
"""
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from dataPretreatment.preprocessText import preprocess_text
from sklearn.cluster import DBSCAN

def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()

def tfIDF(sentences):
    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1)
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
    # 查看特征大小
    print('Features length: ' + str(len(word)))
    return weight



if __name__ == '__main__':
    data = pd.read_csv('../dataset/BV1b7411W7DH.csv')
    sentences = []
    preprocess_text(data['dm_text'].to_list(), sentences)
    random.shuffle(sentences)
    for sentence in sentences[:10]:
        print(sentence)
    weight = tfIDF(sentences)
    print(len(weight))
    print(weight)
    futureData = []
    for i in range(12,13):
        numClass = i  # 聚类分几簇
        clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  # 这里也可以选择随机初始化init="random"
        pca = PCA(n_components=10)  # 降维
        TnewData = pca.fit_transform(weight)  # 载入N维
        s = clf.fit(TnewData)
        #scores = metrics.silhouette_score(TnewData,clf.labels_)
        #scores = np.sqrt(clf.inertia_)
        scores = silhouette_score(weight,clf.labels_,metric='euclidean')
        print(scores)
        print('PCA与TSEN降维')
        newData = PCA(n_components=4).fit_transform(weight)  # 载入N维
        newData = TSNE(2).fit_transform(newData)
        result = list(clf.predict(TnewData))
        plot_cluster(result, newData, numClass)
        data['class'] = result
        data.to_csv('../proData/BV1b7411W7DH.csv',index = False)


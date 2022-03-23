import random
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing
from sklearn import metrics
from dataPretreatment import preprocessText
from dataPretreatment.preprocessText import stopwordslist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def preprocess_text(content_lines, sentences):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in stopwordslist(), segs))  # 去掉停用词
            sentences.append(" ".join(segs))
        except Exception:
            print(line)
            continue

def tfIDF(sentences):
    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
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

def evalCluter(x):
    print()

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

if __name__ == '__main__':
    data = pd.read_csv('dataset/疫情爆发期1.csv',encoding='utf-8')
    sentences = []
    preprocess_text(data['text'].to_list(), sentences)
    random.shuffle(sentences)
    for sentence in sentences[:10]:
        print(sentence)
    weight = tfIDF(sentences)
    for i in range(2,31):
        numClass = i  # 聚类分几簇
        clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  # 这里也可以选择随机初始化init="random"
        pca = PCA(n_components=10)  # 降维
        TnewData = pca.fit_transform(weight)  # 载入N维
        s = clf.fit(TnewData)
        #scores = metrics.silhouette_score(TnewData,clf.labels_)
        scores = np.sqrt(clf.inertia_)

        print(scores)
    # print('PCA降维')
    # pca = PCA(n_components=2)  # 输出两维
    # newData = pca.fit_transform(weight)  # 载入N维
    # result = list(clf.predict(TnewData))
    # plot_cluster(result, newData, numClass)
    #
    # print('TSNE降维')
    #
    # ts = TSNE(2)
    # newData = ts.fit_transform(weight)
    # result = list(clf.predict(TnewData))
    # plot_cluster(result, newData, numClass)
    #
    # from sklearn.manifold import TSNE

    # print('PCA与TSEN降维')
    # newData = PCA(n_components=4).fit_transform(weight)  # 载入N维
    # newData = TSNE(2).fit_transform(newData)
    # result = list(clf.predict(TnewData))
    # plot_cluster(result, newData, numClass)

    # dataset['class'] = result
    # dataset.to_csv('proData/pro疫情潜伏期.csv',index = False)
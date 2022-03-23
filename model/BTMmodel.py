# _*_ coding: utf-8 _*_
"""
Time:     2022/1/12 11:29
Author:   ChenXin
Version:  V 0.1
File:     BTMmodel.py
Describe:  Github link: https://github.com/Chen-X666
"""
import bitermplus as btm
import numpy as np
import pandas as pd
import tmplot

import dataPretreatment.preprocessText as segmentWord
import altair as alt
import pickle as pkl
import tmplot as tmp
import glob


def evalModel(model,T):
    perplexity = model.perplexity_
    coherence = sum(model.coherence_)/len(model.coherence_)
    #entropy = btm.entropy(model.matrix_topics_words_)
    evalScore = perplexity/coherence
    print(perplexity)
    print(coherence)
    return T,perplexity,coherence,evalScore


def fitModel(texts,stop_words,T):
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts, stop_words=stop_words)
    # Vectorizing documents
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    docs_lens = list(map(len, docs_vec))
    # Generating biterms
    biterms = btm.get_biterms(docs_vec)
    # INITIALIZING AND RUNNING MODEL
    model = btm.BTM(
        X, vocabulary, seed=12321, T=T, M=5, alpha=50/T, beta=0.01)
    model.fit(biterms, iterations=1000)

    a = btm.get_top_topic_words(model=model, words_num=5)
    print(a)
    with open("modelData/BTMmodel"+str(T-1)+".pkl", "wb") as file:
        pkl.dump(model, file)
    file.close()
    print(model.matrix_topics_words_)

    return model



def visualization(model,texts):
    # Reading documents from a file
    docs = pd.read_csv('dataset/疫情潜伏期.csv', header=None).values.ravel()

    # Plotting topics as a scatter plot
    topics_coords = tmp.prepare_coords(model)
    tmp.plot_scatter_topics(topics_coords, size_col='size', label_col='label')

    # # Plotting terms probabilities
    # terms_probs = tmp.calc_terms_probs_ratio(phi, topic=0, lambda_=1)
    # tmp.plot_terms(terms_probs)

    # Running report interface
    tmp.report(model, docs=docs, width=250)

def filterStableTopics(filesPath):
    # Loading saved models
    models_files = sorted(glob.glob(filesPath))
    models = []
    for fn in models_files:
        file = open(fn, 'rb')
        models.append(pkl.load(file))
        file.close()


    # Choosing reference model
    np.random.seed(122334)
    reference_model = np.random.randint(1, 6)

    # Getting close topics
    close_topics, close_kl = tmp.get_closest_topics(
        models, method="sklb", ref=reference_model)
    print(close_kl)
    print(close_topics)
    # Getting stable topics
    stable_topics, stable_kl = tmp.get_stable_topics(
        close_topics, close_kl, ref=reference_model, thres=0.7)
    print(stable_topics)
    # Stable topics indices list
    print(stable_topics[:, reference_model])


if __name__ == '__main__':
    texts = []
    df = pd.read_csv(
        'dataset/疫情潜伏期.csv',encoding='utf-8'
    )['text'].to_list()
    for s in df:
        texts.append(segmentWord.seg_depart(s))
    # df = pd.read_csv(
    #     'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
    # texts = df['texts'].str.strip().tolist()
    stop_words = segmentWord.stopwordslist()
    evalData = []
    for i in range(3,4):
        model = fitModel(texts=texts,stop_words=stop_words,T=i)
        evalData.append(evalModel(model,i))
        visualization(model,texts=texts)


    evalData = pd.DataFrame(columns=['topic','perplexity', 'coherence', 'evalScore'], data=evalData)  # 数据有三列，列名分别为one,two,three
    evalData.to_csv('evalData/Eval潜伏期BTMmodel.csv', encoding='utf-8', index=False)
    #filterStableTopics(filesPath='C:\\Users\\Chen\\Desktop\\bulletProjects\\TopicModel\\model\\modelData\\BTMmodel[0-9].pkl')




    #p_zd = model.transform(docs_vec)
    #print(btm.perplexity(model.matrix_topics_words_, p_zd, X, 100))
    # vis = pyLDAvis.prepare(
    # model.matrix_words_topics_, topics, np.count_nonzero(X, axis=1), vocabulary, np.sum(X, axis=0))
    # pyLDAvis.save_html(vis, 'online_btm.html')  # path to output
    # p_zd = model.transform(docs_vec)
    #perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
    #
    # a = tmp.report(model=model, docs=texts)
    # from ipywidgets.embed import embed_minimal_html
    # embed_minimal_html('export.html', views=[a], title='Widgets export')

    #
    # #IMPORTING DATA
    # df = pd.read_csv(
    #     'dataset/test.txt', header=None, names=['texts'])
    # texts = df['texts'].str.strip().tolist()
    # # PREPROCESSING
    # # Obtaining terms frequency in a sparse matrix and corpus vocabulary
    # stop_words = ["十分"]
    # X, vocabulary, vocab_dict = btm.get_words_freqs(texts, stop_words=stop_words)
    # print(X, vocabulary, vocab_dict)
    # print(len(vocabulary))
    # tf = np.array(X.sum(axis=0)).ravel()
    # print(tf)
    # # Vectorizing documents
    # docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    # docs_lens = list(map(len, docs_vec))
    # # Generating biterms
    # biterms = btm.get_biterms(docs_vec)
    # print(biterms)
    # # INITIALIZING AND RUNNING MODEL
    # model = btm.BTM(
    #     X, vocabulary, seed=12321, T=8, M=5, alpha=50/8, beta=0.01)
    #
    #
    #
    #
    # model.fit(biterms, iterations=20)
    # p_zd = model.transform(docs_vec)
    # print(model.df_words_topics_)
    #
    #
    # # METRICS
    # perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
    # print(perplexity)
    # coherence = btm.coherence(model.matrix_topics_words_, X, M=2)
    #
    # a = tmp.report(model=model, docs=texts)
    # from ipywidgets.embed import embed_minimal_html
    # embed_minimal_html('export.html', views=[a], title='Widgets export')



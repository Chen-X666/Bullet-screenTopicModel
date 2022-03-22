# _*_ coding: utf-8 _*_
"""
Time:     2022/1/5 20:23
Author:   ChenXin
Version:  V 0.1
File:     Main.py
Describe:  Github link: https://github.com/Chen-X666
"""
#coding:utf-8
"""
-model: Specify the topic model LDA,DMM、
        LDA: Specify the Latent Dirichlet Allocation topic model
        DMM: Specify the one-topic-per-document Dirichlet Multinomial Mixture model
        BTM: Infer topics for Biterm
        WNTM: Infer topics for WNTM
        SATM: Infer topics using SATM
        PTM: Infer topics using PTM
        GPUDMM: Infer topics using GPUDMM
        GPU_PDMM: Infer topics using GPU_PDMM
        LFLDA: Infer topics using LFLDA
        LFDMM: Infer topics using LFDMM
        LDAinf: Infer topics for unseen corpus using a pre-trained LDA model
        DMMinf: Infer topics for unseen corpus using a pre-trained DMM model
        Eval: Specify the document clustering evaluation
-corpus: Specify the path to the input corpus file.
-vectors: Specify the path to the word2vec file.
-ntopics <int>: Specify the number of topics. The default value is 20.
-alpha <double>: Specify the hyper-parameter alpha. Following [2, 8], the default alpha value is 0.1.
-beta <double>: Specify the hyper-parameter beta. The default beta value is 0.01 which is a common setting in literature.
-niters <int>: Specify the number of Gibbs sampling iterations. The default value is 1000.
-twords <int>: Specify the number of the most probable topical words. The default value is 20.
-name <String>: Specify a name to the topic modeling experiment. The default value is model.
-sstep <int>: Specify a step to save the sampling outputs. The default value is 0 (i.e. only saving the output from the last sample).
"""
from sklearn.model_selection import GridSearchCV  # 网格搜索
import subprocess
import os

def runCmd(cmd) :
        res = subprocess.Popen(cmd, shell=True,  stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        sout ,serr = res.communicate() #该方法和子进程交互，返回一个包含 输出和错误的元组，如果对应参数没有设置的，则无法返回
        return res.returncode, sout, serr, res.pid #可获得返回码、输出、错误、进程号；

if __name__ == '__main__':
    #'GPUDMM', 'GPU_PDMM', 'LFLDA', 'LFDMM'
    models = ['BTM', 'WNTM','LDA', 'DMM', 'SATM', 'PTM']
    #models = ['BTM']
    for model in models:
        print(model)
        cmdString = 'java -jar STTM.jar -model {model} -corpus {corpus} -ntopics {ntopics} ' \
                    '-alpha {alpha} -beta {beta} -niters {niters} -twords {twords} -name {name} -sstep {sstep}'\
            .format(model=model,corpus='dataset/BV1m7411F7si.csv.txt',ntopics=8,alpha=0.1,beta=0.01,niters=2000,twords=10,name=model,sstep=0)
        res = runCmd(cmdString)
        print(res[0])
        print(res[1])
        print(res[2])
        print(res[3])
    #从元组中获得对应数据


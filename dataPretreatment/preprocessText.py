# _*_ coding: utf-8 _*_
"""
Time:     2022/1/2 18:52
Author:   ChenXin
Version:  V 0.1
File:     preprocessText.py
Describe:  Github link: https://github.com/Chen-X666
"""

# 创建停用词列表
import jieba


def stopwordslist():
    stopwords = [line.strip() for line in open(r'C:\Users\Chen\Desktop\bulletProjects\TopicModel\dataPretreatment\data\stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行中文分词去停用词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence)
    result = ' '.join(sentence_depart)
    # 创建一个停用词列表
    #stopwords = stopwordslist()
    # 输出结果为outstr
    # 去停用词
    # words = []
    # for word in sentence_depart:
    #     outstr = ''
    #     print(word)
    #     if word != '\t':
    #         outstr = word + " "
    # words.append(outstr)

    return result

def preprocess_text(content_lines, sentences):
    for line in content_lines:
        #if len(line)>4:
            try:
                segs = jieba.lcut(line)
                #segs = [v for v in segs if not str(v).isdigit()]  # 去数字
                segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
                segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
                segs = list(filter(lambda x: x not in stopwordslist(), segs))  # 去掉停用词
                if len(segs) >2:
                    sentences.append(" ".join(segs))
            except Exception:
                print(line)
                continue
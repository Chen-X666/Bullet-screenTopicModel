# _*_ coding: utf-8 _*_
"""
Time:     2022/1/2 17:53
Author:   ChenXin
Version:  V 0.1
File:     tool.py
Describe:  Github link: https://github.com/Chen-X666
"""
import jieba
if __name__ == '__main__':
    a = jieba.lcut('大佬我也借个素材，已三连。人民万岁')
    print(a)

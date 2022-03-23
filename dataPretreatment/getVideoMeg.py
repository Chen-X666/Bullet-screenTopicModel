# _*_ coding: utf-8 _*_
"""
Time:     2022/1/23 20:03
Author:   ChenXin
Version:  V 0.1
File:     getVideoMeg.py
Describe:  Github link: https://github.com/Chen-X666
"""
import asyncio
import time

from bilibili_api import video, Credential
import pandas as pd

async def getVideoMsg(bvid):
    # 实例化 Video 类
    v = video.Video(bvid=bvid)
    # 获取信息
    info = await v.get_info()
    # 打印信息

    return info

if __name__ == '__main__':
    df = pd.read_csv('dataset/疫情潜伏期.csv',encoding='utf-8')
    bvnos = df['bvno'].drop_duplicates().to_list()
    for bvno in bvnos:
        videoMsg = asyncio.get_event_loop().run_until_complete(getVideoMsg(bvid=bvno))
        videoPubdate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(videoMsg["pubdate"]))
        videoTitle = videoMsg['title']
        print(bvno)
        print(videoPubdate)
        print(videoTitle)
        print()
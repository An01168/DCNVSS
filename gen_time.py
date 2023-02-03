import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import json


nonkey_time = 94
key_time = 321
local_atten_time = 9
key_select_time = 7

f = open("../valImages_key.txt","r")
cur = f.readlines()
f.close()

sum_time = 0
count = 0

for i in range(len(cur)):
    pre = cur[i].strip().split(' ')
    num_key = 1
    num_nonkey = len(pre) - 1
    one_time = nonkey_time*num_nonkey + num_key*key_time
    count = count + len(pre)
    sum_time += one_time

time = sum_time / count
print('time:', time)

datas = []
for i in range(len(cur)):
    dic = {}
    pre = cur[i].strip().split(' ')
    if len(pre) == 1:
        pre2 = pre[0]
        Lab2 = pre2.replace('leftImg8bit_sequence', 'gtFine').replace('leftImg8bit', 'gtFine_labelTrainIds')
        dic = dict(height=1024, width=2048, pre_img=pre2, cur_img=pre2,
                   fpath_segm=Lab2.rstrip())
        datas.append(dic)
    else:
        pre2 = pre[0]
        Lab2 = pre2.replace('leftImg8bit_sequence', 'gtFine').replace('leftImg8bit', 'gtFine_labelTrainIds')
        firframe = pre[0].split('_')
        dic = dict(height=1024, width=2048, pre_img=pre2, cur_img=pre2,
                   fpath_segm=Lab2.rstrip())
        datas.append(dic)
        num = len(pre)
        for j in range(1, num):
            num3 = pre[j]
            pre3 = firframe[0] + '_'+ firframe[1]+ '_'+ firframe[2] + '_' + num3+ '_leftImg8bit.png'
            dic = dict(height=1024, width=2048, pre_img=pre3, cur_img=pre2,
                       fpath_segm=Lab2.rstrip())
            datas.append(dic)

with open('./valCity_video_select.odgt','w') as json_file:
    for each_dict in datas:
        json_file.write(json.dumps(each_dict)+'\n')









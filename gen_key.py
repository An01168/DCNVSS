import torch
import torch.nn as nn
import os
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models import ModelBuilder
import torch.nn.functional as F
from models.function import similarFunction, weightingFunction


kH = 9
kW = 9

builder = ModelBuilder()
low_encoder = builder.build_encoder(
    arch= 'low_resnet101dilated',
    fc_dim=2048,
    weights='')
low_encoder = low_encoder.cuda()


nonkey_match = builder.build_match(
    fc_dim= 1024,
    weights='./ckpt/video-PSP101/match_epoch_40.pth',
    kH = kH,
    kW = kW)
nonkey_match = nonkey_match.cuda()


key_select = builder.build_keyselect(
    fc_dim= 1024,
    weights= './ckpt/video-PSP101/keyselect_epoch_20.pth')
key_select = key_select.cuda()


project = '../Cityscapes_video/'
f = open("../valImages.txt","r")
cur = f.readlines()
f.close()

f_weighting = weightingFunction.apply

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

with open('./valImages_key.txt','w') as json_file:
    for i in range(len(cur)):
        datas = []
        #count = 0
        pre = cur[i].split('_')
        pre0 =  pre[0].replace("leftImg8bit","leftImg8bit_sequence")
        num = int(pre[2])
        num0 = str(num).zfill(6)


        pre00 = pre0 + '_'+ pre[1]+ '_'+ num0+ '_leftImg8bit.png'
        datas.append(pre00)

        label0 = pre0.replace("leftImg8bit_sequence","gtFine")
        label = label0 + '_'+ pre[1]+ '_'+ num0+ '_gtFine_labelTrainIds.png'

        img0_dir = project + pre00
        img0_path = os.path.join(img0_dir)

        img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img0 = cv2.resize(img0, (713, 713), interpolation=cv2.INTER_LINEAR)
        img0_input = torch.from_numpy(img0.transpose((2, 0, 1))).float()

        label_dir = project + label
        label_path = os.path.join(label_dir)

        segm = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        segm = cv2.resize(segm, (713, 713), interpolation=cv2.INTER_NEAREST)
        segm = torch.from_numpy(segm.astype(np.int)).long()
        segm = segm.unsqueeze(0).cuda()

        for t1, m1, s1 in zip(img0_input, mean, std):
            t1.sub_(m1).div_(s1)

        img0_input = img0_input.unsqueeze(0)
        nonkey_input = img0_input.cuda()

        for j in range(1, 20):

            num1 = str(num-j).zfill(6)
            pre1 = pre0 + '_'+ pre[1]+ '_'+ num1+ '_leftImg8bit.png'

            img1_dir = project + pre1
            img1_path = os.path.join(img1_dir)

            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (713, 713), interpolation=cv2.INTER_LINEAR)
            img1_input = torch.from_numpy(img1.transpose((2, 0, 1))).float()

            for t1, m1, s1 in zip(img1_input, mean, std):
                t1.sub_(m1).div_(s1)
            img1_input = img1_input.unsqueeze(0).cuda()

            concat_input = torch.cat((img1_input, nonkey_input), 0)
            with torch.no_grad():
                lowfea = low_encoder(concat_input)
                key_lowfea = lowfea[0,:,:,:]
                key_lowfea = key_lowfea.unsqueeze(0)
                nonkey_lowfea = lowfea[1,:,:,:]
                nonkey_lowfea = nonkey_lowfea.unsqueeze(0)
                local_atten = nonkey_match(key_lowfea, nonkey_lowfea)
                key_pred = key_select(key_lowfea, nonkey_lowfea, local_atten)
                key_pred = key_pred[0,0]
                key_pred = 100.0*key_pred


            key_pred = key_pred.data.cpu()
            if key_pred < 11.52:
                datas.append(num1)
                print(num1)
            #else:
            #    break

        for each_dict in datas:
            json_file.write(each_dict+' ')
        json_file.write('\n')

















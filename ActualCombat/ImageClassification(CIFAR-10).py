import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


d2l.DATA_HUB['cifar10_tiny']=(d2l.DATA_URL+'kaggle_cifar10_tiny.zip',
                              '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
demo=True
if demo:
    data_dir=d2l.download_extract('cifar10_tiny')
else:
    data_dir='../data/cifar-10'

#整理数据集
def read_csv_labels(frame):
    '''读取frame来给标签字典返回一个文件名'''
    with open(frame,'r') as f:
        lines=f.readlines()[1:]
    tokens=[l.rstrip().split(',')for l in lines]
    return dict(((name,label)for name,label in tokens))
labels=read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))
print(labels)

def copyfile(filename,target_dir):
    '''将文件名复制到目标目录'''
    os.makedirs(target_dir,exist_ok=True)
    shutil.copy(filename,target_dir)


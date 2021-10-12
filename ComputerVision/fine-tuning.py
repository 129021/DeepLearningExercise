import matplotlib
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

#从网络下载热狗数据
d2l.DATA_HUB['hotdog']=(d2l.DATA_URL+'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
# d2l.DATA_HUB['hotdog']=('../data/hotdog.zip')
data_dir=d2l.download_extract('hotdog')
train_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'))
test_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'))

#图像的大小和纵横比各有不同
hotdogs=[train_imgs[i][0]for i in range(8)]
not_hotdogs=[train_imgs[-i-1][0]for i in range(8)]
d2l.show_images(hotdogs+not_hotdogs,2,8,scale=1.4)
plt.show

#数据增广
normalize=torchvision.transforms.Normalize(
    [0.485,0.456,0.406],[0.229,0.224,0.225]
)
train_augs=torchvision.transforms.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),normalize
])

test_augs=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),normalize
])

#定义和初始化模型
pretrained_net=torchvision.models.resnet18(pretrained=True)
pretrained_net.fc

finetune_net=torchvision.models.resnet18(pretrained=True)
finetune_net.fc=nn.Linear(finetune_net.fc.in_features,2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

#微调模型


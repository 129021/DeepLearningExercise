import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

#下载完整数据集的小规模demo
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')
demo=True
if demo:
    data_dir=d2l.download_extract('dog_tiny')
else:
    data_dir=os.path.join('..','data','dog-breed-identification')

#整理数据集
def reorg_dog_data(data_dir,valid_ratio):
    labels=d2l.read_csv_labels(os.path.join(data_dir,'labels.csv'))
    d2l.reorg_train_valid(data_dir,labels,valid_ratio)
    d2l.reorg_test(data_dir)

batch_size=32 if demo else 128
valid_ratio=0.1
reorg_dog_data(data_dir,valid_ratio)

#图像增广
transform_train=torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224 x 224的新图像
    torchvision.transforms.RandomResizedCrop(
        224,scale=(0.08,1.0),
        ratio=(3.0/4.0,4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(
        brightness=0.4,contrast=0.4,saturation=0.4),
    #增加随机噪声
    torchvision.transforms.ToTensor(),
    #标准化图像的每个通道
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
])

transform_test=torchvision.transforms.Compose([
    torchvision.transformsl.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
])

#读取数据集
train_ds,train_valid_ds=[
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir,'train_valid_test',folder),
        transform=transform_train) for folder in ['train','train_valid']
]

valid_ds,test_ds=[
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir,'train_valid_test',folder),
        transform=transform_test) for folder in ['valid','test']
]

train_iter,train_valid_iter=[
    torch.utils.data.DataLoader(
        dataset,batch_size,shuffle=True,drop_last=True)
    for dataset in (train_ds,train_valid_ds)
]

valid_iter=torch.utils.data.DataLoader(
    valid_ds,batch_size,shuffle=False,drop_last=True
)

test_iter=torch.utils.data.DataLoader(test_ds,batch_size,shuffle=False,drop_last=False)

#微调预训练模型
def get_net(devices):
    finetune_net=nn.Sequential()
    finetune_net.features=torchvision.models.resnet34(pretrained=True)
    finetune_net.output_new=nn.Sequential(
        nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,120)
    )
    finetune_net=finetune_net.to(devices[0])
    for param in  finetune_net.features.parameters():
        param.requires_grad=False
    return finetune_net

#

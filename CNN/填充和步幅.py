import torch
from torch import nn
from d2l import torch as d2l

def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:])
conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1)
X=torch.rand(size=(8,8))
# print(comp_conv2d(conv2d,X).shape)

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
# print(comp_conv2d(conv2d,X).shape)


'''多输入通道互相关运算'''
def corr2d_multi_in(X,K):
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

X=torch.tensor([[[0,1,2],[3,4,5],[6,7,8]],
                [[1,2,3],[4,5,6],[7,8,9]]])
K=torch.tensor([[[0,1],[2,3]],[[1,2],[3,4]]])

# print(corr2d_multi_in(X,K))
'''多通道输入互相关函数'''

def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
K=torch.stack((K,K+1,K+2),0)
# print(K.shape)

# print(corr2d_multi_in_out(X,K))



'''1×1卷积'''
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)
    return Y.reshape((c_o,h,w))
X=torch.normal(0,1,(3,3,3))
K=torch.normal(0,1,(2,3,1,1))
Y1=corr2d_multi_in_out_1x1(X,K)
Y2=corr2d_multi_in_out(X,K)
assert float(torch.abs(Y1-Y2).sum())<1e-6


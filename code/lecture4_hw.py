''''
Homework:
    对给定的数据集，设计一个MLP网络，实现分类任务。
'''

# generate dataset
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn

def dataset_generator(N):
    # center
    c1 = torch.tensor((0,math.sqrt(2)),dtype=torch.float)
    c2 = torch.tensor((-1,-1),dtype=torch.float)
    c3 = torch.tensor((1,-1),dtype=torch.float)
    #
    classes = 3
    samples_perclass = N
    train_data = []
    train_label = []
    for i in range(samples_perclass):
        # random radius sampled from uniform distribution on [0,1]
        r1 = math.sqrt(2)*np.random.uniform()
        theta1 = 2*math.pi*np.random.uniform()
        x1 = [c1[0]+r1*math.cos(theta1),c1[1]+r1*math.sin(theta1)]
        train_data.append(x1)
        train_label.append(0)
        r2 = math.sqrt(2)*np.random.uniform()
        theta2 = 2*math.pi*np.random.uniform()
        x2 = [c2[0]+r2*math.cos(theta2),c2[1]+r2*math.sin(theta2)]
        train_data.append(x2)
        train_label.append(1)
        r3 = math.sqrt(2)*np.random.uniform()
        theta3 = 2*math.pi*np.random.uniform()
        x3 = [c3[0]+r3*math.cos(theta3),c3[1]+r3*math.sin(theta3)]
        train_data.append(x3)
        train_label.append(2)

    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    #
    # fig = plt.figure()
    # plt.scatter(train_data[:,0],train_data[:,1],c=train_label)
    # plt.show()

    return train_data,train_label

def dataloader(N):
    data,label = dataset_generator(N)
    mydst = TensorDataset(data,label)
    loader = DataLoader(dataset=mydst,batch_size=50,shuffle=True)

    return loader

class mlpnet(nn.Module):
    def __init__(self):
        super(mlpnet,self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(2,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,3),
            nn.LogSoftmax(dim=1)
        )
    def forward(self,x):
        return self.arch(x)


def train():
    epoches = 10
    lr = 1e-3
    n_train = 10000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = mlpnet().to(device)
    opt = torch.optim.Adam(net.parameters(),lr=lr)
    train_loader = dataloader(n_train)
    loss_fn = nn.NLLLoss()
    #
    for k in range(epoches):
        right_num = torch.tensor(0.0)
        for idx,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            z = net(x)
            label_pred = torch.argmax(z,dim=1)
            right_num += torch.sum(torch.eq(label_pred,y).float())
            loss = loss_fn(z,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_acc = right_num/(n_train*3)
        print( 'epoch: %d | train_acc: %.2f' % (k+1,train_acc) )

    return net

def test(net):
    n_test = 1000
    test_loader = dataloader(n_test)
    device = 'cpu'
    right_num = torch.tensor(0.0)
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        z = net(x)
        label_pred = torch.argmax(z, dim=1)
        right_num += torch.sum(torch.eq(label_pred, y).float())
    test_acc = right_num/(n_test*3)
    print('test_acc : %.2f' % test_acc)


if __name__=='__main__':
    net = train()
    test(net)




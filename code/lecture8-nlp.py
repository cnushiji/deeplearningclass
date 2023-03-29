import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import math
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt

# define a simple RNN net
class my_rnn_net(nn.Module):
    def __init__(self):
        super(my_rnn_net,self).__init__()
        self.rnn = nn.RNN(input_size=2,hidden_size=10,num_layers=3,nonlinearity='tanh',
                     bias=True,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(in_features=20,out_features=1)

    def forward(self,seq):
        H,hn = self.rnn(seq)
        Y = self.fc(H)

        return Y

# generate input and target
def data_generator():
    seq_len = 100
    batch_size = 20
    input_size = 2
    output_size = 1
    train_num = 1000
    test_num = 100
    std_w1 = 1
    std_w2 = 1
    std_v = 1
    # train dataset
    X_train = np.zeros((train_num,seq_len,input_size))
    Y_train = np.zeros((train_num,seq_len,output_size))
    for i in range(train_num):
        for j in range(1,seq_len):
            X_train[i,j,0] = X_train[i,j-1,0] + math.sin(X_train[i,j-1,0]) + std_w1*np.random.randn(1)
            X_train[i,j,1] = X_train[i,j-1,1] + math.cos(X_train[i,j-1,1]) + std_w2*np.random.randn(1)
            Y_train[i,j,0] = math.sqrt(X_train[i,j,0]**2 + X_train[i,j,0]**2) + std_v*np.random.randn(1)
    train_dst = TensorDataset(torch.tensor(X_train,dtype=torch.float),
                              torch.tensor(Y_train,dtype=torch.float))
    train_loader = DataLoader(dataset=train_dst,batch_size=batch_size,shuffle=True)

    # test dataset
    X_test = np.zeros( (test_num,seq_len,input_size) )
    Y_test = np.zeros( (test_num,seq_len,output_size) )
    for i in range(test_num):
        for j in range(1,seq_len):
            X_test[i,j,0] = X_test[i,j-1,0] + math.sin(X_test[i,j-1,0]) + std_w1*np.random.randn(1)
            X_test[i,j,1] = X_test[i,j-1,1] + math.cos(X_test[i,j-1,1]) + std_w2*np.random.randn(1)
            Y_test[i,j,0] = math.sqrt(X_test[i,j,0]**2 + X_test[i,j,0]**2) + std_v*np.random.randn(1)
    test_dst = TensorDataset(torch.tensor(X_test,dtype=torch.float),
                             torch.tensor(Y_test,dtype=torch.float))
    test_loader = DataLoader(dataset=test_dst,batch_size=10,shuffle=False)

    return train_loader,test_loader

# forward propagate
def train():
    # train setting
    N_epoch = 100
    N_inf = 20  # inference per N_inf epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(logdir='./lecture8_res/logs_lecture8/train_loss')

    # load data
    train_loader,test_loader = data_generator()

    #
    model = my_rnn_net()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    step_count = 0
    for k in range(N_epoch):
        train_batch_loss = torch.tensor(0.0)
        for j,data in enumerate(train_loader):
            #
            x,y_gt = data
            x = x.to(device)
            y_gt = y_gt.to(device)
            y = model(x)
            # backward propagation
            loss = loss_fn(y,y_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            train_batch_loss += loss.item()
            step_count += 1
            writer.add_scalar('train/step_loss',loss,step_count)
        train_batch_loss /= len(train_loader)
        print('epoch {} | loss {}'.format(k+1,train_batch_loss))

        # inference
        loss_inf = torch.tensor(0.0).to(device)
        if (k+1)%N_inf==0:
            for i, data in enumerate(test_loader):
                x, y_gt = data
                x = x.to(device)
                y_gt = y_gt.to(device)
                y = model(x)
                loss_inf += loss_fn(y, y_gt)
            loss_inf = loss_inf/len(test_loader)
            writer.add_scalar('test/batch_loss',loss_inf,k+1)
            # visualize
            y = y.detach().numpy()
            y_gt = y_gt.detach().numpy()
            fig,axs = plt.subplots(3,3,figsize=(12,12))
            t = torch.linspace(0,99,100)
            for i in range(9):
                ax = axs[int(i/3),i%3]
                ax.plot(t,y[i],color='r',linestyle='-',label='inf')
                ax.plot(t,y_gt[i],color='b',linestyle='-.',label='gt')
                ax.legend(loc='upper left')
            plt.tight_layout()
            plt.suptitle('epoch-'+str(k+1))
            filename = os.path.join(r'.\lecture8_res','epoch-'+str(k+1)+'.png')
            plt.savefig(filename)
            plt.show()
    #
    writer.close()

#
if __name__ == '__main__':
    train()



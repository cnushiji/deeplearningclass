# compare different optimizers for 1 dimension case
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

# define the model architecture: one hidden layer FCN net
class Mymodel(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):
        super(Mymodel,self).__init__()
        self.netarch = nn.Sequential(
            nn.Linear(inputdim,hiddendim),
            nn.Sigmoid(),
            nn.Linear(hiddendim,outputdim),
        )
    def forward(self,x):
        return self.netarch(x)

# define the target function
def fun(x):
    y = np.sin(x)
    return y

# define data loader
def Mydataloader(N):
    x = np.random.uniform(-np.pi,np.pi,(N,1))
    y = fun(x) + 0.1*np.random.randn(N,1)
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    mydataset = Data.TensorDataset(x,y)
    data_loader = Data.DataLoader(mydataset,batch_size=10,shuffle=True)
    return data_loader

# define the loss function
def loss_criterion(modelout,target):
    criterion = nn.MSELoss()
    return criterion(modelout,target)

# train the model
def model_train():
    num_epoch = 50
    num_train = 1000  # total training samples
    # train data
    train_loader = Mydataloader(num_train)
    # initialize networks
    net_sgd = Mymodel(1,10,1)
    net_momentum = Mymodel(1,10,1)
    net_rmsprop = Mymodel(1,10,1)
    net_adam = Mymodel(1,10,1)
    nets = [net_sgd,net_momentum,net_rmsprop,net_adam]
    # optimizers
    opt_sgd = torch.optim.SGD(net_sgd.parameters(),lr=0.01)
    opt_momentum = torch.optim.SGD(net_momentum.parameters(),lr=0.01,momentum=0.9)
    opt_rmsprop = torch.optim.RMSprop(net_rmsprop.parameters(),lr=0.01,alpha=0.9)
    opt_adam = torch.optim.Adam(net_adam.parameters(),lr=0.01,betas=(0.9,0.999))
    opts = [opt_sgd,opt_momentum,opt_rmsprop,opt_adam]
    # training
    losses = [[],[],[],[]] # loss history of different optimizer
    for epoch in range(num_epoch):
        for k,(xdata,ydata) in enumerate(train_loader):
            for net,opt,loss in zip(nets,opts,losses):
                model_out = net(xdata)
                batch_loss = loss_criterion(model_out,ydata)
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                loss.append(batch_loss.item())
        print('Epoch %d' % epoch)

    # show loss of different optimizers
    plt.figure()
    label = ['SGD', 'Momentum', 'RMSProp', 'Adam']
    for k,loss in enumerate(losses):
        plt.plot(loss,label=label[k])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0,1))
    plt.show()

    return nets

# test the model
def model_test(net):
    num_test = 100 # test sample number
    x = np.linspace(-1,1,num_test).reshape(num_test,1)
    y = fun(x)
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)

    # test loss
    out = net(x)
    out = out.detach()
    test_loss = loss_criterion(out,y)
    test_loss = test_loss.detach().numpy()
    print('Average loss %.4f for %d test samples' % (test_loss,num_test))

    # visual result
    plt.figure()
    plt.plot(x,y,'r')
    plt.scatter(x,out,c='b')
    plt.title('test result')
    plt.legend(['true','out'],loc='lower right',fontsize=10)
    plt.show()

#
if __name__ == "__main__":
    net = model_train()
    #model_test(net)
    print('Optimizer compare for 1d example finished')





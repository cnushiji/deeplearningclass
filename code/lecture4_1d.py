# optimization example for 1 dimension case
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
    y = np.power(x,3) - x + 0*np.random.normal(loc=0.0,scale=0.1,size=x.shape)
    return y

# define data loader
def Mydataloader(N,batchsize=100):
    x = np.random.uniform(-1.0,1.0,(N,1))
    y = fun(x)
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    mydataset = Data.TensorDataset(x,y)
    data_loader = Data.DataLoader(mydataset,batch_size=batchsize,shuffle=True)
    return data_loader

# define the loss function
def loss_criterion(modelout,target):
    criterion = nn.MSELoss()
    return criterion(modelout,target)

# train the model
def model_train():
    num_epoch = 100
    num_train = 1000  # total training samples
    # train data
    train_loader = Mydataloader(num_train,1)
    # initialize network
    net = Mymodel(1,10,1)
    # define the optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    #
    loss = np.zeros((num_epoch,1),dtype=np.float)
    for epoch in range(num_epoch):
        epoch_loss = torch.tensor(0.0)
        for k,(xdata,ydata) in enumerate(train_loader):
            model_out = net(xdata)
            batch_loss = loss_criterion(model_out,ydata)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss = epoch_loss/num_train
        loss[epoch] = epoch_loss
        print('Epoch %d | loss %.4f' % (epoch,epoch_loss.numpy()))

    #
    plt.figure()
    plt.plot(np.arange(num_epoch),loss,label='train epoch loss')

    return net

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
    model_test(net)
    print('example for 1d case finished')





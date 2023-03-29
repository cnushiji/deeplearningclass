# optimization example for 2 dimension case
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# define the model architecture: one hidden layer FCN net
class Mymodel(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):
        super(Mymodel,self).__init__()
        self.netarch = nn.Sequential(
            nn.Linear(inputdim,hiddendim),
            nn.ReLU(),
            nn.Linear(hiddendim,hiddendim),
            nn.ReLU(),
            nn.Linear(hiddendim,outputdim),
        )
    def forward(self,x):
        return self.netarch(x)

# define the target function: 2 dimension
def fun(x):
    # y = np.sin(x[:,0])*np.cos(x[:,1])
    y = np.power(x[:,0],2) + np.power(x[:,1],2)
    y = y.reshape(-1,1)
    return y

# convert vector to 2d matrix
def vec2matrix(z,row,col):
    Z = np.zeros((row,col),dtype=np.float)
    for i in range(row):
        Z[i] = z[i*col:(i+1)*col].reshape(col,)
    return Z

# define data loader
def Mydataloader(N,batchsize=10):
    x1 = np.linspace(-2 * np.pi, 2 * np.pi,N)
    x2 = np.linspace(-2 * np.pi, 2 * np.pi,N)
    X1,X2 = np.meshgrid(x1,x2)  # generate coordinate matrix
    xy = np.hstack([X1.flatten().reshape(-1,1),X2.flatten().reshape(-1,1)])
    z = fun(xy)
    xy = torch.tensor(xy,dtype=torch.float32)
    z = torch.tensor(z,dtype=torch.float32)
    mydataset = Data.TensorDataset(xy,z)
    data_loader = Data.DataLoader(mydataset,batch_size=batchsize,shuffle=True)
    return data_loader

# define the loss function
def loss_criterion(modelout,target):
    criterion = nn.MSELoss()
    return criterion(modelout,target)

# train the model
def model_train():
    num_epoch = 10
    num_train = 100  # total training samples
    # train data
    train_loader = Mydataloader(num_train)
    # initialize network
    net = Mymodel(2,10,1)
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
        print('Epoch %d | loss %.4f' % (epoch,epoch_loss.numpy()) )
    #
    plt.figure()
    plt.plot(np.arange(num_epoch),loss,label='train epoch loss')
    plt.show()

    return net

# test the model
def model_test(net):
    num_test = 100 # test sample number
    x = np.sort(np.random.uniform(-3,3,(1,num_test)))
    y = np.sort(np.random.uniform(-3,3,(1,num_test)))
    X,Y = np.meshgrid(x,y)
    xy = np.hstack((X.flatten().reshape(-1,1),Y.flatten().reshape(-1,1)))
    z = fun(xy)
    xy = torch.tensor(xy,dtype=torch.float32)
    z = torch.tensor(z,dtype=torch.float32)

    # test loss
    out = net(xy)
    out = out.detach()
    test_loss = loss_criterion(out,z)
    test_loss = test_loss.detach().numpy()
    print('Average loss %.4f for %d test samples' % (test_loss,num_test))

    # visual result
    f = plt.figure()
    ax = Axes3D(f)
    Z = vec2matrix(z,num_test,num_test)
    model_out = vec2matrix(out,num_test,num_test)
    ax.plot_surface(X,Y,Z-model_out,cmap='rainbow')
    ax.scatter(xy[:,0],xy[:,1],out,c='b',marker='.')
    plt.show()

#
if __name__ == "__main__":
    net = model_train()
    model_test(net)
    print('example for 2d case finished')





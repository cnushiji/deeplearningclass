# compare different optimization method for mnist figures
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.utils.data as Data
import numpy as np

# define the network class
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1), # indim=28*28
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,padding=1), # indim=14*14
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32,kernel_size=3,padding=1), # indim=7*7
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),   # in: batch*32*3*3 filters, out: batch*(32*9)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*3*3,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=10),
        )
    # forward propagation
    def forward(self,x):
        x = self.conv(x)
        return self.fc(x)

# define mnist image data loader
def Mydataloader(batchsize=64):
    trainset = datasets.MNIST(root=r'D:\research\openprojects\Datasets\MNIST\mnist-py',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)
    testset = datasets.MNIST(root=r'D:\research\openprojects\Datasets\MNIST\mnist-py',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=False)
    train_loader = Data.DataLoader(dataset=trainset,batch_size=batchsize,shuffle=True)
    test_loader = Data.DataLoader(dataset=testset,batch_size=batchsize,shuffle=False)
    return train_loader,test_loader

# define the models and optimizers
def Models_and_Optimizers(lr):
    # models
    net_sgd = Mymodel()
    net_rmsprop = Mymodel()
    net_adam = Mymodel()
    nets = [net_sgd,net_rmsprop,net_adam]

    # optimizers
    opt_sgd = torch.optim.SGD(net_sgd.parameters(),lr=lr)
    opt_rmsprop = torch.optim.RMSprop(net_rmsprop.parameters(),lr=lr)
    opt_adam = torch.optim.Adam(net_adam.parameters(),lr=lr)
    opts = [opt_sgd,opt_rmsprop,opt_adam]

    return nets,opts

# train the model
def model_train():
    # settings
    num_epoch = 1
    lr = 1e-3
    losses = [[],[],[]] # losses of 3 different optimizers
    #
    train_loader,_ = Mydataloader()
    nets,opts = Models_and_Optimizers(lr)
    loss_fn = nn.CrossEntropyLoss()
    #
    for epoch in range(num_epoch):
        for k,(x,label) in enumerate(train_loader):
            for net,opt,loss in zip(nets,opts,losses):
                out = net(x)
                batch_loss = loss_fn(out,label)
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                loss.append(batch_loss.item())
                # batch accuracy
                batch_acc = torch.mean((torch.eq(torch.argmax(out,dim=1),label)).float())
            print('Epoch %d | Batch %d | batch_loss %.4f | Acc %.2f ' % \
                                (epoch,k,batch_loss.item(),batch_acc))
        print('Epoch %d training finished' % epoch)
    # show
    plt.figure()
    labels = ['SGD', 'RMSProp', 'Adam']
    for k in range(len(labels)):
        plt.plot(losses[k],label=labels[k])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0,3))
    plt.show()

    return nets

# test the model
def model_test(nets):
    #
    _,test_loader = Mydataloader()
    loss_fn = nn.CrossEntropyLoss()
    #
    test_losses = np.zeros((3,))
    test_acc = np.zeros(3,)
    for k,net in enumerate(nets):
        net.eval()
        right_num = torch.tensor(0.0)
        testnum = 10000
        for i,(x,labels) in enumerate(test_loader):
            out = net(x)
            batch_loss = loss_fn(out,labels)
            class_predict = torch.argmax(out,dim=1)
            right_num += torch.sum(torch.eq(class_predict,labels).float())
            test_losses[k] += batch_loss.item()
        # calculate test set accuracy for net[k]
        test_acc[k] = right_num.item()/testnum
        test_losses[k] = test_losses[k]/len(test_loader)
    #
    print("acc_sgd: %.4f | acc_rmsprop: %.4f | acc_adam: %.4f" %
          (test_acc[0], test_acc[1], test_acc[2]))
    print("loss_sgd: %.4f | loss_rmsprop: %.4f | loss_adam: %.4f" %
          (test_losses[0],test_losses[1],test_losses[2]))
    #
    result_visualize(x,class_predict,labels)

# result visualize
def result_visualize(x,label_pred,label_gt):
    '''
    :param x: N*C*H*W
    :param label_pred: N,
    :param label_gt: N,
    :return:
    '''
    n = x.size(0)
    if n>16:
        n=16
    class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(4,4,sharex=True,sharey=True)
    fig.set_size_inches(10,10)
    for i in range(n):
        img = x[i].permute(1,2,0)
        img_label_pred = label_pred[i]
        img_label_gt = label_gt[i]
        classname_pred = class_names[img_label_pred]
        classname_gt = class_names[img_label_gt]
        ax = axs[int(i/4),i%4]
        ax.imshow(img)
        ax.set_title('p:'+classname_pred+'/g:'+classname_gt)
    plt.tight_layout()
    plt.show()
#
if __name__ == "__main__":
    nets = model_train()
    model_test(nets)
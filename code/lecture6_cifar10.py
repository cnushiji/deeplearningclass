'''
    设计一个残差网络来实现CIFAR10数据集的分类
    CIFAR10数据集情况：
        一共10类(机/汽车/鸟/猫/鹿/狗/袜/马/船/卡车)，6万张彩色图片，图片尺寸32*32
        训练集：每类5000张
        测试集：每类1000张
'''
import os
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import TensorDataset,DataLoader
import pickle
import glob
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
import tqdm
import time

#
def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def data_loader(dict,batch_size):
    xdata = dict[b'data'] # N*3072 row-wise stack, r,g,b
    xdata = torch.from_numpy(xdata.astype(np.float32) / 255).reshape(-1, 3, 32, 32)
    label = torch.tensor(dict[b'labels'],dtype=torch.long)  # b indicates the string behind which is of 'bytes' type
    mydataset = TensorDataset(xdata,label)
    data_loader = DataLoader(dataset=mydataset,batch_size=batch_size,shuffle=True)

    return data_loader
# define a residual block
class residual_block(nn.Module):
    def __init__(self,c_in=3,c_h=64):
        super(residual_block,self).__init__()
        self.block_arch = nn.Sequential(
            nn.Conv2d(in_channels=c_in,out_channels=c_h,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_h,out_channels=c_in,kernel_size=3,stride=1,padding=1))

    def forward(self,x):
        y = self.block_arch(x)
        z = x + y
        z = F.relu(z)
        return z

class mymodel(nn.Module):
    def __init__(self,c_in=3):
        super(mymodel,self).__init__()
        self.convhead = nn.Conv2d(in_channels=c_in,out_channels=64,kernel_size=3,padding=1)
        self.resblock1 = residual_block(c_in=64,c_h=64)
        self.resblock2 = residual_block(c_in=64,c_h=128)
        self.convtail = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,padding=1)
        self.fcblock = nn.Sequential(
            nn.Linear(in_features=4096,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            # nn.Softmax()
        )

    def forward(self,x):
        #
        y = self.convhead(x)
        #
        y1 = self.resblock1(y)
        y1 = F.max_pool2d(y1, kernel_size=2, stride=2)
        #
        y2 = self.resblock2(y1)
        y2 = F.max_pool2d(y2, kernel_size=2, stride=2)
        #
        y3 = self.convtail(y2)
        y3 = F.max_pool2d(y3, kernel_size=2, stride=2) # out: N*256*4*4
        # fully connected block
        # y4 = torch.flatten
        y4 = y3.view(y3.size(0),-1)
        z = self.fcblock(y4)

        return z

def train(src_path):
    #
    train_file_paths = glob.glob(os.path.join(src_path, 'data_*'))
    test_file_path = os.path.join(src_path, 'test_batch')
    save_dir = './lecture6_res/checkpoints'
    #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epoch_num = 100
    batch_size = 64
    savestep = 20
    lr = 1e-4
    model = mymodel(c_in=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss() # include softmax already
    writer = SummaryWriter('./lecture6_res/runs/losses/')
    #
    train_acc = []
    for k in range(epoch_num):
        # train
        for i in range(len(train_file_paths)):
            ''' batch file'''
            file_path = train_file_paths[i]
            print(file_path)
            dict = unpickle(file_path)
            train_data_loader = data_loader(dict,batch_size)
            for l,(x,label) in enumerate(train_data_loader):
                x = x.to(device)
                label = label.to(device)
                y = model(x)
                loss = loss_fn(y,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # batch accuracy calculation
                class_pred = torch.argmax(y,dim=1)
                train_batch_acc = torch.sum(class_pred==label)/x.size(0)
                train_acc.append(train_batch_acc)
                #
                writer.add_scalar('Accuracy/train_batchstep',
                                  scalar_value=train_batch_acc,
                                  global_step=k*len(train_file_paths)*len(train_data_loader)+i*len(train_data_loader)+l+1)
                writer.add_scalar('loss/train_batchstep',
                                  scalar_value=loss,
                                  global_step=k*len(train_file_paths)*len(train_data_loader)+i*len(train_data_loader)+l+1)
                print('epcoch {} | batchfile {} | step {} | loss {} | acc {} '.format(k, i, l, loss.item(),train_batch_acc))
        # test
        if (k+1) % savestep == 0:
            # save model
            state_dict = {'net_state_dict': model.state_dict()}
            torch.save(state_dict, os.path.join(save_dir, 'resnet-epoch-' + str(k + 1) + '-modelpara.pth'))
            # test
            test_acc = test(test_file_path,model,device=device)
            writer.add_scalar('Accuracy/test',scalar_value=test_acc,global_step=int((k+1)/savestep) )

    writer.close()

    return model

# test
def test(test_file_path,model=None,batch_size=128,device=None,visualize=False):
    #
    if model==None:
        model = mymodel(c_in=3)
        state_dict = torch.load('./lecture6_res/checkpoints/resnet-epoch-60-modelpara.pth',
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['net_state_dict'])
    model.eval() # only works when there are BN and Dropout
    #
    print(test_file_path)
    dict = unpickle(test_file_path)
    test_data_loader = data_loader(dict,batch_size)
    correct_num = 0
    test_samples = 1e4
    for i,(x,label) in enumerate(test_data_loader):
        print('test batch %d' % i)
        x = x.to(device)
        label = label.to(device)
        y = model(x)
        class_predict = torch.argmax(y,dim=1)
        correct_num += torch.sum(class_predict==label)
    acc = correct_num/test_samples
    print('test set accuracy %.2f' % acc.item())
    # local visualize
    if visualize==True:
        result_visualize(x,class_predict,label)

    return acc

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
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
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

if __name__=='__main__':
    # server
    # src_path = './cifar-10-batches-py'
    # train(src_path)

    # local test
    src_path = r'D:\research\openprojects\Datasets\CIFAR10\cifar-10-batches-py'
    test_file_path = os.path.join(src_path, 'test_batch')
    test(test_file_path,visualize=True)





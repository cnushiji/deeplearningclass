# section 8.2 基于GAN生成图像
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 定义超参数
image_size = 784
num_epoches = 20
batch_size = 128
lr = 0.0002

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,),std=(0.5,))])   # 3 for RGB channels
dataset = torchvision.datasets.MNIST(root=r'D:\research\openprojects\Datasets\MNIST\mnist-py',
                                     train=True,transform=transform,download=False)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 构建CGAN模型，主要包括生成器G和判别器D
# 构建判别器
class Discriminator(nn.Module):
    def __init__(self,image_size=784,num_labels=10,emd_size=10):
        super(Discriminator,self).__init__()
        self.label_emd = nn.Embedding(num_labels,emd_size)
        self.model = nn.Sequential(
            nn.Linear(image_size+emd_size, 1024),nn.LeakyReLU(0.2,inplace=True),nn.Dropout(0.4),
            nn.Linear(1024, 512),nn.LeakyReLU(0.2,inplace=True),nn.Dropout(0.4),
            nn.Linear(512, 256),nn.LeakyReLU(0.2,inplace=True),nn.Dropout(0.4),
            nn.Linear(256,1),
            nn.Sigmoid())

    def forward(self,x,labels):
        x = x.view(x.size(0),784)
        c = self.label_emd(labels)
        x = torch.cat([x,c],dim=1)
        out = self.model(x)
        return out.squeeze()

D = Discriminator().to(device)

# 构建生成器

class Generator(nn.Module):
    def __init__(self,latent_size = 110,num_labels=10,emd_size=10):
        super().__init__()
        self.label_emd = nn.Embedding(num_labels,emd_size)
        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512), nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,784), nn.Tanh())

    def forward(self,z,labels):
        z = z.view(z.size(0),100)
        c = self.label_emd(labels)
        x = torch.cat([z,c],dim=1)
        out = self.model(x)
        return out.view(x.size(0),28,28)

G = Generator().to(device)

# 优化器
criterion = nn.BCELoss()
D_optimizer = torch.optim.Adam(D.parameters(),lr=lr)
G_optimizer = torch.optim.Adam(G.parameters(),lr=lr)

# 定义函数
def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def reset_grad():
    D_optimizer.zero_grad()
    G_optimizer.zero_grad()

# 训练网络
sample_dir = r'C:\Users\shiji\Desktop\dlclass\code\DCGAN'
total_step = len(data_loader)
writer = SummaryWriter(log_dir='logs')
for epoch in range(num_epoches):
    for i,(images,labels) in enumerate(data_loader):
        # ===============================================
        #                训练判别器
        # ===============================================
        images = images.to(device)
        labels = labels.to(device)

        # 定义图像真或假的标签
        real_labels = torch.ones(images.size(0),).to(device)
        fake_labels = torch.zeros(images.size(0),).to(device)

        # 对真图像的损失函数
        real_validity = D(images,labels)
        labels_float = labels.float()
        d_loss_real = criterion(real_validity, real_labels)

        # 对假图像的损失函数
        z = torch.randn(images.size(0),100).to(device)
        gen_labels = torch.randint(0,10,(images.size(0),)).to(device)
        fake_images = G(z,gen_labels)
        fake_validity = D(fake_images,gen_labels)
        d_loss_fake = criterion(fake_validity,fake_labels)

        # 判别器总的损失函数
        d_loss = d_loss_real + d_loss_fake

        # 梯度清零
        reset_grad()
        d_loss.backward()
        D_optimizer.step()
        # ===============================================
        #                训练生成器
        # ===============================================
        # 定义生成器对于假图像的损失函数
        z = torch.randn(images.size(0),100).to(device)
        gen_labels = torch.randint(0,10,(images.size(0),)).to(device)
        fake_images = G(z,gen_labels)
        outputs = D(fake_images,gen_labels)

        # 生成器的损失函数
        g_loss = criterion(outputs, real_labels)

        #
        reset_grad()
        g_loss.backward()
        G_optimizer.step()

        if (i+1) % 200==0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss:{:.4f}'\
                  .format(epoch,num_epoches,i+1,total_step,d_loss.item(),g_loss.item()))

    # 保存损失函数
    writer.add_scalars('scalars', {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}, epoch + 1)

    # 保存真图像
    if (epoch+1)==1:
        images = images.reshape(images.size(0),1,28,28)
        save_image(denorm(images),os.path.join(sample_dir,'real_images.png'))
    # 保存假图像
    fake_images = fake_images.reshape(fake_images.size(0),1,28,28)
    save_image(denorm(fake_images),os.path.join(sample_dir,'fake_images-{}.png'.format(epoch+1)))

# 可视化结果
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
reconsPath = './CGAN/fake_images-10.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)
plt.axis('off')
plt.show()

from torchvision.utils import make_grid
z = torch.randn(100,100).to(device)
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)
images = G(z,labels).unsqueeze(1)
grid = make_grid(images,nrow=10,normalize=True)
fig,ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1,2,0).detach().cpu().numpy(),cmap='binary')
ax.axis('off')
plt.show()

# 指定标签的数据生成
def generate_digit(generator,digit):
    z = torch.randn(1,100).to(device)
    label = torch.LongTensor([digit]).to(device)
    img = generator(z,label).detach().cpu()
    img = 0.5*img + 0.5
    return transforms.ToPILImage()(img)

img = generate_digit(G,8)
img.show()
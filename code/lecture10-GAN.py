# section 8.2 基于GAN生成图像
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

# 定义超参数
image_size = 784
num_epoches = 10
batch_size = 128
lr = 0.0002
latent_size = 100
hidden_size = 256
# 数据预处理
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,),std=(0.5,))])   # 3 for RGB channels
dataset = torchvision.datasets.MNIST(root=r'D:\research\openprojects\Datasets\MNIST\mnist-py',
                                     train=True,transform=transform,download=False)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 构建GAN模型，主要包括生成器G和判别器D
# 构建判别器
Discriminator = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())
D = Discriminator.to(device)
# 构建生成器
Generator = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
G = Generator.to(device)

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
sample_dir = r'C:\Users\shiji\Desktop\dlclass\code\GAN'
total_step = len(data_loader)
for epoch in range(num_epoches):
    for i,(images,_) in enumerate(data_loader):
        images = images.reshape(images.size(0),-1).to(device)
        # 定义图像真或假的标签
        real_labels = torch.ones(images.size(0),1).to(device)
        fake_labels = torch.zeros(images.size(0),1).to(device)
        # ===============================================
        #                训练判别器
        # ===============================================
        # 对真图像的损失函数
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_scores = outputs
        # 对假图像的损失函数
        z = torch.randn(images.size(0),latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_scores = outputs
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
        z = torch.randn(images.size(0),latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        reset_grad()
        g_loss.backward()
        G_optimizer.step()

        if (i+1) % 200==0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss:{:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch,num_epoches,i+1,
                total_step,d_loss.item(),g_loss.item(),real_scores.mean().item(),
                fake_scores.mean().item()))

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
reconsPath = './GAN/fake_images-10.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)
plt.axis('off')
plt.show()
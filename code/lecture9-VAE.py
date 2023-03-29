# section 8.1 基于变分自编码器生成图像
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

# 定义超参数
image_size = 784
h_dim = 400
z_dim = 20
num_epoches = 20
batch_size = 128
lr = 0.001

# 数据预处理
dataset = torchvision.datasets.MNIST(root=r'D:\research\openprojects\Datasets\MNIST\mnist-py',
                                     train=True,transform=transforms.ToTensor(),download=False)
data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

# 构建模型
# 定义VAE模型，主要有encoder和decoder构成
class VAE(nn.Module):
    def __init__(self,image_size=784,h_dim=400,z_dim=20):
        super(VAE,self).__init__()
        self.fc1 = nn.Linear(image_size,h_dim)
        self.fc2 = nn.Linear(h_dim,z_dim)
        self.fc3 = nn.Linear(h_dim,z_dim)
        self.fc4 = nn.Linear(z_dim,h_dim)
        self.fc5 = nn.Linear(h_dim,image_size)

    def encode(self,x):
        h = F.relu(self.fc1(x))
        return self.fc2(h),self.fc3(h)

    def reparameterize(self,mu,log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self,x):
        mu,log_var = self.encode(x)
        z = self.reparameterize(mu,log_var)
        x_reconst = self.decode(z)
        return x_reconst,mu,log_var

# 选择计算设备和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

# 训练模型
model.train()
for epoch in range(num_epoches):
    train_loss = 0.0
    i = 0
    for imgs, labels in data_loader:
        imgs = imgs.to(device)
        x = imgs.view(imgs.size(0), -1)
        x_reconst, mu, log_var = model(x)
        # 损失函数
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconst_loss + kl_div
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示损失值
        train_loss += loss.item()
        if i % 100 == 99:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, train_loss / 2000))
            train_loss = 0.0
        i += 1

# 测试模型
save_dir = r'D:\research\openprojects\book-Python based on PyTorch\chapt8_GenerativeModel\vae_res'
with torch.no_grad():
    # 保存采样图像
    z = torch.randn(batch_size, z_dim).to(device)
    out = model.decode(z).view(-1, 1, 28, 28)
    save_image(out, os.path.join(save_dir, 'reconst-hidden.png'))
    # 保存原始图像和重构图像的对比
    examples = enumerate(data_loader)
    batch_id,(imgs,labels) = next(examples)
    imgs = imgs.to(device)
    x = imgs.view(imgs.size(0), -1)
    out, _, _ = model(x)
    x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
    save_image(x_concat, os.path.join(save_dir, 'reconst-test.png'))

# 展示原图像及重构图像
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
reconsPath = 'reconst-test.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)
plt.axis('off')
plt.show()



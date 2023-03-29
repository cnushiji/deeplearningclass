import numpy as np
import torch

# x = torch.tensor(2.0,requires_grad=True)
# y = torch.pow(x,2)
# print(x.requires_grad)
# y.backward()
# print(x.grad)
#
# x = torch.tensor(np.pi/3,requires_grad=True)
# y2 = torch.sin(x)
# y2.backward()
# print(x.grad)

x3 = torch.tensor(1.0)
w = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(1.0,requires_grad=True)
y3 = w*x3 + b
y3.backward()

w.grad.data.zero_()
x4 = torch.tensor(2.5)
y4 = w*x4 + b
y4.backward()
print(w.grad)
print(w.is_leaf)
print(y4.is_leaf)


'''
    alexnet = models.vgg19(pretrained=True)
    print(alexnet.features[0].weight)
    这些模型要求输入的图片至少是224的，3通道，0~1范围
'''

from torchvision import transforms
import torchvision.models as models
import PIL.Image as Image

#
img_path = r'C:\Users\shiji\Desktop\杂项\others\lena.png'
img = Image.open(img_path)
f = transforms.ToTensor()
img_in = f(img)

#
resnet18 = models.resnet18(pretrained=True)
for i in resnet18.state_dict():
    print(i)

# clf_result = resnet18(img_in)

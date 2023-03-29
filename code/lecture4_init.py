# initialization
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MLPnet(nn.Module):
    def __init__(self,x_dim,h_dim1,h_dim2,y_dim):
        super(MLPnet, self).__init__()
        self.arch = nn.Sequential(
            nn.Linear(x_dim, h_dim1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(h_dim1, h_dim2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(h_dim2,y_dim),
            nn.Sigmoid()
        )
        # initialization method 1
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        return self.arch(x)

# define a weight initialization function
def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)

    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='leaky_relu')
        # bias initialization
        # nn.init.uniform_(m.bias,-1,1)

    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

# instantiate a class
net = MLPnet(x_dim=2,h_dim1=10,h_dim2=10,y_dim=2)
# initialization method 2
net.apply(weight_init)
print(net)

if __name__=='__main__':
    x = torch.rand(100,2)
    y = net(x)
    y = y.detach().numpy()

    fig = plt.figure()
    plt.plot(x[:,0],x[:,1],'r-.')
    plt.plot(y[:, 0], y[:, 1], 'b-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()









import torch
import torchvision
import torchvision.transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('train_test_datasets', train=False, transform=torchvision.transforms.ToTensor()
                                       , download=False)
# 注意调用函数时要加括号。
dataload = DataLoader(dataset, batch_size=64)


# 创建神经网络

class hedy(nn.Module):
    def __init__(self):
        super(hedy, self).__init__()
        # 开始写各种操作

        self.relu = nn.Sigmoid()

    def forward(self, x_1):
        y_1= self.relu(x_1)
        return y_1



# 可视化数据
writer = SummaryWriter("logs")
# 实例化神经网络
nn_mode = hedy()
# 遍历数据
step = 0
for data in dataload:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    img_mode = nn_mode.forward(imgs)
    writer.add_images("output", img_mode, step)
    step += 1
writer.close()

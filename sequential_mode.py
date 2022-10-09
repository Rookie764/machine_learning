import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("train_test_datasets", train=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)


class hedy(nn.Module):
    def __init__(self):
        super(hedy, self).__init__()
        self.mode = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        y = self.mode(x)
        return y


loss_crossentryp = nn.CrossEntropyLoss()
Hedy = hedy()
# 利用GPU训练
Hedy = Hedy.cuda()
loss_crossentryp=loss_crossentryp.cuda()
# 创建一个优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
Optimizer = torch.optim.SGD(Hedy.parameters(), lr=0.007)
for epoch in range(5):
    total_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        imgs=imgs.cuda()
        targets=targets.cuda()
        output_img = Hedy.forward(imgs)  # 将输入送入神经网络训练
        loss = loss_crossentryp(output_img, targets)  # 求的损失值
        Optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播求的相应的梯度
        Optimizer.step()  # 一步一步进行优化
        total_loss += loss
    print(total_loss)

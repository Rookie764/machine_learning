import torch

loss = torch.nn.L1Loss()  # 直接计算误差相加
loss_mse = torch.nn.MSELoss()  # 计算误差的平方和
loss_entropy = torch.nn.CrossEntropyLoss()  # 交叉熵误差计算和,loss(x,class)=-class+ln(exp(x))
# 其中x是代表类别的向量，class是要计算的某一个类别

input = torch.randn(1, 5)  # 有五个类别
a1 = torch.empty(1, dtype=torch.long).random_(5)  # 取一个

output = loss_entropy(input, a1)
print(input, a1)
print(output)

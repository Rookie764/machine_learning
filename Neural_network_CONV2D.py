import torch

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernal = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input_1 = input.resize(1, 1, 5, 5)  # 查看官方文档的说明，要有四个参数，原本的tensor数据只有行列两个数据。
kernal_1 = kernal.resize(1, 1, 3, 3)

output = torch.nn.functional.conv2d(input_1, kernal_1, stride=1)  # 查看官方文档的示例。
print(output)
output_2=torch.nn.functional.conv2d(input_1,kernal_1,stride=2)
print(output_2)
output_3=torch.nn.functional.conv2d(input_1,kernal_1,stride=1,padding=1)
print(output_3)
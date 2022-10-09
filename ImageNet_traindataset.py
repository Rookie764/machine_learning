import torch
import torchvision
import os
import torch.nn as nn
os.environ['TORCH_HOME']='D:/pythonProject/torch-model'

# train_dataset=torchvision.datasets.ImageNet("Image_test",transform=torchvision.transforms.ToTensor(),
#                                             download=True,train=True)
# vgg_false=torchvision.models.vgg16(pretrained=False)
# vgg_true=torchvision.models.vgg16(pretrained=True)
#
# vgg_true.classifier.add_module("7",nn.Linear(1000,10)) #在下载的模型中添加环节
# print(vgg_false)
# vgg_false.classifier[6]=nn.Linear(4096,10)
# print(vgg_false)

#先导入模型

vgg16=torchvision.models.vgg16(pretrained=False)

torch.save(vgg16,"vgg16_method1.pth")

torch.save(vgg16.state_dict(),"vgg16_method2.pth")
import torchvision
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
dataset_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 将数据集转换成tensor形式

dataset_train = torchvision.datasets.CIFAR10("./train_test_datasets", train=True,transform=dataset_trans, download=True)
dataset_test = torchvision.datasets.CIFAR10("./train_test_datasets", train=False, transform=dataset_trans,download=True)

writer=SummaryWriter('logs')

img, target = dataset_train[0]  # 用这个去获得对象
# print(img)
writer.add_image("dataset_train",img,0)
writer.close()

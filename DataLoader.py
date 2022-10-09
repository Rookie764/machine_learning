import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_test=torchvision.datasets.CIFAR10("train_test_datasets",train=False,transform=torchvision.transforms.ToTensor())

dataset_loader=DataLoader(dataset_test,batch_size=64,shuffle=True,num_workers=0,drop_last=False)



writer=SummaryWriter("Dataload_test")
for epcho in range(2):
    step=0
    for data in dataset_loader:
        imgs,targets=data
        writer.add_images("{}".format(epcho),imgs,step)
        step+=1
writer.close()

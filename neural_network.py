from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

    def forward(self,inpout):
        ouput=inpout*2
        return ouput
if __name__ == '__main__':

    model=NN()
    x=torch.tensor(2.0)
    y=model.forward(x)
    print(y)

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy
import torchmetrics
import pandas as pd
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning.loggers as pll

all=[True,False][0]
#dir(pll)
if all:
    for key, value in pll.__dict__.items():
        print('=====================')
        print(key)
        print(value)
        f = open('loggers_info/AVAILABLE.txt', "a")
        if "AVAILABLE" in key and True:
            f.write('=================\n')
            f.write(key.split('AVAILABLE')[0]+'\n')
            f.write(str(value)+'\n')

        if 'Logger' in key and type(value)!=bool:
            #help(value)
            f = open('loggers_info/'+key+".txt", "a")
            #if type(help(value))==str:
            f.write(str(value)+'\n')
            f.write('Here is the list of contents\n')
            for i,j in value.__dict__.items():
                f.write(str(i)+'\n')
                f.write(str(j)+'\n')


            f.close()
#            if 'function' in help(value):
 #               help(value)

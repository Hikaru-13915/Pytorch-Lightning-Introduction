from __future__ import print_function
import argparse

import os
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy
import torchmetrics
import pandas as pd
import pytorch_lightning.loggers as pll

all=[True,False][0]
#dir(pll)
if all:
    dir_path='loggers_info'
    if dir_path not in os.listdir('.'):
        os.mkdir(dir_path)
    for key, value in pll.__dict__.items():
        print('=====================')
        print('')
        print(key)
        print(value)

        f = open('loggers_info/AVAILABLE.txt', "a")
        if "AVAILABLE" in key and True:
            f.write('=================\n')
            f.write(key.split('AVAILABLE')[0]+'\n')
            f.write(str(value)+'\n')
        tag=False
        if 'Logger' in key and type(value)!=bool and tag:
            f = open('loggers_info/'+key+".txt", "a")
            f.write(str(value)+'\n')
            f.write('Here is the list of contents\n')
            for i,j in value.__dict__.items():
                f.write(str(i)+'\n')
                f.write(str(j)+'\n')
            f.close()

#call
from pytorch_lightning.callbacks import Callback


class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print("Starting to initialize the trainer!")

    def on_init_end(self, trainer):
        print("trainer is initialized now")

    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")
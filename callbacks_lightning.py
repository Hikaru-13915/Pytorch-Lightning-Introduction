from __future__ import print_function
import argparse

import os
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy
import torchmetrics
import pandas as pd
import pytorch_lightning.callbacks as cbs

all=[True,False][0]
#dir(pll)
if all:
    dir_path='callbacks_info'
    if dir_path not in os.listdir('.'):
        os.mkdir(dir_path)
    cb_list = cbs.__all__
    for key, value in cbs.__dict__.items():
        print('=====================')
        print('')
        files=os.listdir('./callbacks_info/')
        f = open('callbacks_info/AVAILABLE.txt', "a")
        if key in cb_list:
            if 'callbacks_info/AVAILABLE.txt' not in files:
                f.write('=================\n')
                f.write(str(key)+'\n')
            f2 = open('callbacks_info/'+key+".txt", "a")
            if key+'.txt' not in files:
                f2.write(str(value)+'\n')
                f2.write('Here is the list of contents\n')
                f2.write(' \n')
                for i,j in value.__dict__.items():
                    f2.write(str(i)+'\n')
                    print(j)
                    try:
                        f2.write(str(j)+'\n')
                    except UnicodeEncodeError:
                        pass
                f2.close()
#call
from pytorch_lightning.callbacks import Callback


class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print("Starting to initialize the trainer!")

    def on_init_end(self, trainer):
        print("trainer is initialized now")

    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")
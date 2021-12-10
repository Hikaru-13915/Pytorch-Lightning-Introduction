from __future__ import print_function
import argparse
from pytorch_lightning import callbacks
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
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.callbacks import ModelCheckpoint

print('=============================================================')
message="This is a scrpit to save a model using callback function called check-point"
print(message)
print('=============================================================')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu1=nn.ReLU(inplace=True)
        self.relu2=nn.ReLU(inplace=True)
        self.relu3=nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
        
class LitAutoEncoder(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        #Set the target Network
        self.model=Net()

        #Set the calculation method of accuracy
        self.accuracy = torchmetrics.Accuracy()
        self.batch_size=args.batch_size
        #Set the target dataset
        self.dataset=MNIST

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model.forward(x)
    
    def setup(self, stage):
        self.mnist_test = self.dataset(os.getcwd(), download=True, train=False, transform=transforms.ToTensor())
        mnist_full = self.dataset(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # It is independent of forward
        x, y = batch
        x_hat = self(x)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log("train_acc", self.accuracy(x_hat, y), prog_bar=False)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        x_label = torch.argmax(x_hat, dim=1)
        acc = torch.sum(y == x_label) * 1.0 / len(x)
        self.log('test_loss', loss)
        self.log("test_acc", self.accuracy(x_hat, y), prog_bar=False)
        results = {'test_loss': loss, 'test_acc': acc}
        return results
    
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.accuracy(x_hat, y), prog_bar=True)
        #return {'x_hat': x_hat, 'y': y, 'batch_loss':loss.item()*x.size(0)}
        return {"valid_loss": loss, 'valid_acc': self.accuracy(x_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['valid_acc'] for x in outputs]).mean()
        results = {'valid_loss': avg_loss, 'valid_acc': avg_acc}
        self.log("valid_loss", avg_loss, prog_bar=False)
        self.log("valid_acc", avg_acc, prog_bar=True)
        
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def main():
# Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the current Model')
    parser.add_argument('--named_modules', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--named_parameters', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load_model', type=str, default="",
                        help='For Loading the current Model')
                        
    args = parser.parse_args()
#End of setting
    #Setting a target model which is written by LightningModule
    autoencoder = LitAutoEncoder(args)
    #Saving parameters before training
    before_train=autoencoder.named_parameters()
    #Custumising a callback function to treat things about check_point
    dir_path='check_points'
    if dir_path not in os.listdir('.'):
        os.mkdir(dir_path)
    check_point_callbacks=ModelCheckpoint(
        #Quantity to monitor, the last epoch will be saved
        monitor="valid_loss",
        #model path to save your trained model
        dirpath="check_points",
        #The name of check point
        filename="mnist-{epoch:02d}-{valid_loss:.2f}",
        #The number of models to save. *Top quality
        save_top_k=3,
        #If you want to save only weights of a model, just tyr 'save_weights_only=True'
    )
    #Trainer setting: You can regard this is a kind of an observer for the training
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        logger=CSVLogger(save_dir="logs/", name="mnist_train"),
        callbacks=check_point_callbacks
        )
    trainer.fit(autoencoder)
#    trainer.save_checkpoint(os.path.join(output,))
    print("===========================")
    print("This is the path of the best checkpoint")
    print(check_point_callbacks.best_model_path)
    print("===========================")
    #If you want to reset the trainer, declare 'trainer=pl.Trainer($callback_name$=False)'
    #In this script, 'check_point_callbacks' corresponds to $callback_name$
    
 

if __name__ == '__main__':
    main()
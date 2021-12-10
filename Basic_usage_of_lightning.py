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
import torchmetrics
import pandas as pd
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt


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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
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
    # Iteration-wise process for test
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
    # Epoch-wise process
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

# End of setting
    autoencoder = LitAutoEncoder(args)
    if args.named_modules:
        print('===modules===')
        for name, module in autoencoder.model.named_modules():
            print(name, type(module))
    if args.named_parameters:
        print('===params===')
        for name, params in autoencoder.named_parameters():
            print(name)#, params)
    
    #when you have already trained a model and want to use the model
    if args.load_model=='':
    
        before_train=autoencoder.named_parameters()

        #gpus=AVAIL_GPUS
        trainer = pl.Trainer(
            gpus=1,#gpu=-1 to use all available gpus
            max_epochs=args.epochs,
            logger=CSVLogger(save_dir="logs/", name="mnist_train")
            )


        trainer.fit(autoencoder)

        ##You can also fit a model by giving date loader directly for training and validation like the three lines bellow.
        #dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
        #train, val, test = random_split(dataset, [55000, 4500,500])    
        #trainer.fit(autoencoder, DataLoader(train,batch_size=64), DataLoader(val,batch_size=64), DataLoader(test,batch_size=64))
        
        if args.named_parameters:
            print('===params===')
            for name, params in autoencoder.named_parameters():
                print(name, params)
        after_train=autoencoder.named_parameters()
        if before_train==after_train:
            print('not.trained!!!')
        else:
            print('trained!!!')
        if args.save_model:
            torch.save(autoencoder.state_dict(), "mnist_lightning.pt")
    else:
        autoencoder.load_state_dict(torch.load('mnist_lightning.pt'))
        
#test step
    if args.test:
        trainer.test()#logger=CSVLogger(save_dir="logs/", name="mnist_test"))
    
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    print("===metrics.head():train===")
    print(metrics)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    plt.figure()
    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "valid_loss"]].plot(grid=True, legend=True, title="Loss over "+str(args.epochs)+' epochs',xlabel="epochs",ylabel="Loss")
    plt.savefig('loss.pdf')
    plt.savefig('loss.png')
    plt.close('all')

    plt.figure()
    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_acc", "valid_acc"]].plot(grid=True, legend=True, title="Accuracy over "+str(args.epochs)+' epochs',xlabel="epochs",ylabel="Accuracy")
    plt.savefig('acc.pdf')
    plt.savefig('acc.png')
    plt.close('all')
    
    print(df_metrics)

if __name__ == '__main__':
    main()

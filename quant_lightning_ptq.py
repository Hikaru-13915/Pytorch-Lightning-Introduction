from __future__ import print_function
import argparse
import torch
from torch._C import device
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
from torch.quantization import QuantStub, DeQuantStub


#Imoprting a model from other src
from models import LitAutoEncoder, Net, QNet

print('=============================================================')
message="This is a scrpit to create a quantized model using normal Torch-ptq(Post Training Quantization) out of a model pretrained with PyTorch lightning"
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

class QNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model=Net()
        self.quant=QuantStub
        self.dequant=DeQuantStub
        self.batch_size=args.batch_size
        
    def forward(self, x):
        x=self.quant(x)
        x=self.model.forward(x)
        x=self.dequant(x)
        return x

    def setup(self, stage):
        self.dataset=MNIST
        self.accuracy = torchmetrics.Accuracy()
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
        x_hat = self.model.forward(x)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log("train_acc", self.accuracy(x_hat, y), prog_bar=False)
        return loss
    # iteration-wise process for test
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
    
    # epoch-wise process
    def test_end(self, outputs):
        #torch.stack appends every params to a list of the first one 
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model.forward(x)
        loss = F.cross_entropy(x_hat,y)
        
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.accuracy(x_hat, y), prog_bar=True)
        return {'x_hat': x_hat, 'y': y, 'batch_loss':loss.item()*x.size(0)}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def save_models(model, params, modules, name):
    torch.save(model. state_dict(), 'Mnist_'+name+'.pt')
    files=os.listdir('./convert_info/')
    draft=name+"_params.txt"
    if draft not in files:
        f = open('convert_info/'+draft, "a")
        f.write(name+'\n')
        f.write('\n')
        f.write("=====================\n")
        f.write('===params===\n')            
        for i,j in params:
            f.write(i+' '*(10-len(i)))
            f.write(str(type(j))+' '*(20-len(str(type(j))))+'\n')
            f.write(str(j)+'\n')
            f.write('---------------------------\n')
    draft=name+"_modules.txt"
    if not draft in files:
        f = open('convert_info/'+draft, "a")
        f.write(name+'\n')
        f.write('\n')
        f.write("=====================\n")
        f.write('===modules===\n')
        for i, j in modules:
            f.write(i+' '*(10-len(i)))
            f.write(str(type(j))+' '*(30-len(str(type(j))))+': ')
            f.write(str(j)+'\n')
            f.write('---------------------------')
    return model.named_parameters(), model.named_parameters() 



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


    autoencoder = LitAutoEncoder(args)
    autoencoder.load_state_dict(torch.load('mnist_lightning.pt'))
    
    if args.named_modules or args.named_parameters:
        print("=====================")
        print("=Before Quantization=")
    if args.named_modules:
        print('===modules===')
        for name, module in autoencoder.model.named_modules():
            print(name, type(module))
    if args.named_parameters:
        print('===params===')
        for name, params in autoencoder.named_parameters():
            print(name)#, params)
    before_ppr=save_models(autoencoder,autoencoder.named_parameters(),autoencoder.named_modules(), "Not prepared")
    trainer = pl.Trainer(gpus=1,max_epochs=args.epochs,logger=CSVLogger(save_dir="logs/", name="mnist_train"))
    if args.test:    #trainer.fit(autoencoder, DataLoader(train,batch_size=64), DataLoader(val,batch_size=64), DataLoader(test,batch_size=64))
        trainer.test()#logger=CSVLogger(save_dir="logs/", name="mnist_test"))
            
    qmodel=QNet(args).to("cpu")
    
    qmodel.load_state_dict(torch.load('mnist_lightning.pt'))
    
    qmodel.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    for i in qmodel.named_parameters():
        print(i[0])
    #qmodel_fp32_fused = torch.quantization.fuse_modules(qmodel.model, [['conv1', 'conv2']])
    qmodel_fp32_prepared = torch.quantization.prepare(qmodel)
    qmodel_int8 = torch.quantization.convert(qmodel_fp32_prepared)

    before_qat=save_models(qmodel_fp32_prepared,qmodel_fp32_prepared.named_parameters(),qmodel_fp32_prepared.named_modules(), "Prepared")
    after_qat=save_models(qmodel_int8,qmodel_int8.model.named_parameters(),qmodel_int8.named_modules(), "After converted")
    trainer=pl.Trainer()
    test_loss = 0
    correct = 0
    test_loader=DataLoader(MNIST(os.getcwd(), download=True, train=False, transform=transforms.ToTensor()), batch_size=args.batch_size)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device="cpu"
    #qmodel_int8.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = qmodel_int8(data)#log function should be outside of Net
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    #量子化後のモデルのテストは、普通のPytorchの書き方でできるかな？
    trainer.test(qmodel)
    if args.named_modules or args.named_parameters:
        print("=====================")
        print("=After Quantization=")
    if args.named_modules:
        print('===modules===')
        for name, module in autoencoder.model.named_modules():
            print(name, type(module))
    if args.named_parameters:
        print('===params===')
        for name, params in autoencoder.named_parameters():
            print(name)#, params)
    #trainer.test(autoencoder)
    

    if before_qat==after_qat:
        print('not quantized!!!!!!!!!!!!!!!!!')
    else:
        print('quantized!!!!!!!!!!!!!!!!!')
    
    print('=============')
    print('model size')
    print('=============')
    print('before prepared')
    print(os.path.getsize('mnist_lightning_qat_not_prepared.pt'))
    print('=============')
    print('after_prepared')
    print(os.path.getsize('mnist_lightning_qat_after_prepared.pt'))
    print('=============')
    print('converted')
    print(os.path.getsize('mnist_lightning_qat_converted.pt'))

if __name__ == '__main__':
    main()

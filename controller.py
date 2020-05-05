import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.transforms as transforms

import argparse
import numpy as np
import time
import datetime
import os
import logging

from models.resnet import *
from models.resnet_att_v4 import *
from models.resnet_avg import *
from models.mvcnn import *
from models.mvcnn_att import *
import util
from logger import Logger
from custom_dataset import MultiViewDataSet

np.set_printoptions(suppress=True,precision=2)

MVCNN = 'mvcnn'
RESNET = 'resnet'
RESNET_ATT = 'resnet_att'
MVCNN_ATT='mvcnn_att'
MODELS = [RESNET,RESNET_ATT,MVCNN,MVCNN_ATT]
START=str(datetime.datetime.now())

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET))
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading data')

input_size=224
train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.25,.25,.25], [3.98, 3.98, 3.98])
        ])
val_transforms =  transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([.25,.25,.25], [3.98, 3.98, 3.98])
        ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset


                                 


dset_train = MultiViewDataSet(args.data, 'train', transform=train_transforms)
n0,n1,n2=dset_train.counts['0'],dset_train.counts['1'],dset_train.counts['2']
train_sampler = torch.utils.data.sampler.WeightedRandomSampler([float(n0+n1+n2)/n0]*n0 +[float(n0+n1+n2)/n1]*n1 + [float(n0+n1+n2)/n2]*n2 , (n0+n1+n2))
train_loader = DataLoader(dset_train, sampler=train_sampler,batch_size=args.batch_size, num_workers=28,pin_memory=True)
eval_train_loader=DataLoader(dset_train,batch_size=32, num_workers=28,pin_memory=True)
print('train size',len(train_loader))
dset_val = MultiViewDataSet(args.data, 'val', transform=val_transforms)
val_sampler=torch.utils.data.sampler.WeightedRandomSampler( [1]*128,128,replacement=False)
#val_loader = DataLoader(dset_val, shuffle=True, batch_size=args.batch_size, num_workers=28,pin_memory=True)
eval_val_loader=DataLoader(dset_val,batch_size=32,num_workers=28,pin_memory=True)
#print('val size',len(val_loader))
classes = dset_train.classes


if args.model == RESNET:
    if args.depth == 18:
        model = resnet18(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 34:
        model = resnet34(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 50:
        model = resnet50(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 101:
        model = resnet101(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 152:
        model = resnet152(pretrained=args.pretrained, num_classes=len(classes))
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
elif args.model == RESNET_ATT:
    if args.depth == 18:
        model = resnet18_att(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 34:
        model = resnet34_att(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 50:
        model = resnet50_att(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 101:
        model = resnet101_att(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 152:
        model = resnet152_att(pretrained=args.pretrained, num_classes=len(classes))
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
elif args.model=='mvcnn':
    model = mvcnn(pretrained=args.pretrained,num_classes=len(classes))
    print('Using ' + args.model)

else:
    model = mvcnn_att(pretrained=args.pretrained,num_classes=len(classes))
    print('Using ' + args.model)

model.to(device)
cudnn.benchmark = True

# print('Running on ' + str(device))

logger = Logger('logs/'+START)

# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0





# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'

    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train():
    train_size = len(train_loader)
    s=time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs).cuda(device)
        # inputs=torch.stack(inputs,1).cuda(device)
        targets = targets.cuda(device)
        #inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = model(inputs)
        #loss=cross entropy + attention entropy
        # loss = criterion(outputs, targets) 
        loss = criterion(outputs, targets) #- 1e-3*torch.mean(model.attention * torch.log(model.attention + 1e-5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % args.print_freq == 0:
            t=time.time() - s 
            print("Iter [%d/%d] Loss: %.6f time taken: %.3f samples per sec: %.3f  time per iteration: %.3f" % (i + 1, train_size, loss.item(), t ,args.batch_size*args.print_freq/t,t/args.print_freq))
            #print('attention weights: {}'.format(model.attention[0,:].data.cpu().numpy()))
            plt.imshow(model.attention[0,:,:].data.cpu().numpy(),aspect=10.0/512,vmax=1,vmin=0)
            plt.pause(0.0001)
            s=time.time()

# Validation and Testing 
def eval(data_loader, is_test=False):
    if is_test:
        load_checkpoint()

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    size = len(data_loader)
    s=time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            #with torch.no_grad():
                # Convert from list of 3D to 4D
            
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs).cuda(device)

            targets = targets.cuda(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

            if (i+1)%args.print_freq==0:
                t=time.time()-s
                print('eval loop %d/%d time %.2f'%(i+1,size,t))
                #print('attention weights: {}'.format(model.attention[0,:].data.cpu().numpy()))
                plt.imshow(model.attention[0,:,:].data.cpu().numpy(),aspect=10.0/512,vmax=1,vmin=0)
                plt.pause(0.0001)
                s=time.time()
    avg_test_acc = 100.0 * float(correct) / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


# Training / Eval loop
if args.resume:
    load_checkpoint()

for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    model.train()
    train()
    print('Time taken: %.2f sec.' % (time.time() - start))
    model.eval()
    print('eval mode')
    # avg_train_acc, avg_loss_train = eval(eval_train_loader)

    # print('\nEvaluation:')
    # print('\tTrain Acc: %.2f - Loss: %.4f' % (avg_train_acc.item(), avg_loss_train.item()))

    avg_val_acc, avg_loss_val = eval(eval_val_loader)

    print('\nEvaluation:')
    print('\tVal Acc: %.2f - Loss: %.4f' % (avg_val_acc, avg_loss_val))
    # print('\tVal Acc: %.2f - Loss: %.4f' % (avg_val_acc.item(), avg_loss_val.item()))

    print('\tCurrent best val acc: %.2f' % best_acc)

    # Log epoch to tensorboard
    # See log using: tensorboard --logdir='logs' --port=6006
    #util.logEpoch(logger, model, epoch + 1, avg_loss_val, avg_val_acc,avg_loss_train,avg_train_acc)
    util.logEpoch(logger, model, epoch + 1, avg_loss_val, avg_val_acc)

    # Save model
    if avg_val_acc > best_acc:
        print('\tSaving checkpoint - Acc: %.2f' % avg_val_acc)
        best_acc = avg_val_acc
        best_loss = avg_loss_val
        util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': avg_val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, args.model, START,str(args.depth))

    # Decaying Learning Rate
    if (epoch + 1) % args.lr_decay_freq == 0:
        lr *= args.lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print('Learning rate:', lr)

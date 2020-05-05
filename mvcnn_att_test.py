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
from models.mvcnn import *
# from models.mvcnn_att import *
# from models.resnet_att import *
# from models.resnet_att_v2 import *
from models.resnet_att_v3 import *


import util
from logger import Logger
from custom_dataset import MultiViewDataSet
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix,classification_report


print('Loading data')

input_size=224
val_transforms =  transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([.25,.25,.25], [3.98, 3.98, 3.98])
        ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
data_dir = '/home/raman/Classification/Views_images/'
# data_dir = '/home/raman/Classification/image_data/sensors'
dset_test = MultiViewDataSet(data_dir, 'test', transform=val_transforms)
print(len(dset_test))
test_loader=DataLoader(dset_test,batch_size=128, num_workers=28,pin_memory=True)

# dataset_sizes = {x: len(dset_test[x]) for x in ['test']}
# class_names = image_datasets['test'].classes
print("length of test set",len(dset_test))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device",device)
criterion = nn.CrossEntropyLoss()

# file='checkpoint/resnet_att18_2019-04-14 23:34:17.423813_checkpoint.pth.tar'#Resnet18_Attn
# file='checkpoint/resnet_att18_2019-04-14 19:56:42.175151_checkpoint.pth.tar'
# file='checkpoint/WAPool/resnet_att18_2019-04-07 21:18:38.418682_checkpoint.pth.tar'#Mod_Resnet18_Attn
# file ='checkpoint/APool/resnet18_2019-04-08 16:55:17.401992_checkpoint.pth.tar'   #Resnet18
# file ='checkpoint/resnet_att_2019-04-05 20:50:28.824025_checkpoint.pth.tar' # Resnet34_attn
# file='checkpoint/WAPool/resnet_att18_2019-04-09 14:26:19.450097_checkpoint.pth.20.tar'#Resnet18Attn20
# file = 'checkpoint/resnet_att18_2019-04-15 17:50:57.877686_checkpoint.pth.tar'
# file = 'checkpoint/resnet_att18_2019-04-24 14:13:01.053939_checkpoint.pth.tar'
# file = 'checkpoint/resnet18_2019-04-26 18:57:27.818601_checkpoint.pth.tar'# MVCNN Max Pool
file = 'checkpoint/resnet_att18_2019-04-28 11:47:46.822106_checkpoint.pth.tar'
# model = resnet18(num_classes=3)  
model = resnet18_att(num_classes=3)  
# file='checkpoint/mvcnn_att_checkpoint.pth.tar'#model_name     
# model = mvcnn_att(num_classes=3)

model.load_state_dict(torch.load(file)['state_dict'])
model.to(device)
model.eval()
cudnn.benchmark = True    
best_acc = 0.0
list_of_test_acc = []
list_of_test_loss = []

since = time.time()

running_loss = 0.0
running_corrects = 0
nb_classes = 3
op,ta=[],[]

def get_att(x):
    x = x.transpose(0, 1)
    #print(x.shape)

    # View pool
    view_pool = []
    weights=[]
    for v in x:
        v = model.conv1(v)
        v = model.bn1(v)
        v = model.relu(v)
        v = model.maxpool(v)

        v = model.layer1(v)
        v = model.layer2(v)
        v = model.layer3(v)
        v = model.layer4(v)

        v = model.avgpool(v)
        v = v.view(v.size(0), -1)
        w = model.V(v)

        view_pool.append(v)
        weights.append(w)
    weights=F.softmax(torch.cat(weights,1),1) 
    return weights

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = np.stack(inputs, axis=1)
        inputs = torch.from_numpy(inputs).cuda(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # atts=get_att(inputs)# use this for testing with attention
        # print(atts)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        op.append(preds.cpu().numpy())
        ta.append(labels.cpu().numpy())
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

op=np.hstack(op)
ta=np.hstack(ta)
print('confusion matrix')
print(confusion_matrix(op,ta))
print('classification report')
print(classification_report(op,ta))
acc=(op==ta).sum()/op.shape
print('acccuracy: {}'.format(acc))


# loss = running_loss / len(dset_test)
# acc = running_corrects.double() / len(dset_test)
# list_of_test_acc += [acc]
# list_of_test_loss += [loss]
# time_elapsed = time.time() - since
# print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# print('Best val Acc: {:4f}'.format(acc))
# plt.subplot(1,2,1)
# plt.plot(list_of_test_acc, label='Test acc')
# plt.plot(list_of_test_loss, label='Test loss')
# # plt.plot(list_of_val_acc, label= 'Test acc')
# plt.legend(frameon=False)
# plt.show()

from models.mvcnn import *
from models.mvcnn_att import *
from models.resnet_att import *
#import mvcnn and mvcnn_att modules here

def params(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(params(resnet18_att(num_classes=3)))
print(params(resnet34_att(num_classes=3)))
print(params(resnet50_att(num_classes=3)))
print(params(mvcnn()))
print(params(mvcnn_att()))


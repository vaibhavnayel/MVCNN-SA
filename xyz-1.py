import torchvision.models as models
#import mvcnn and mvcnn_att modules here

def params(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)

params(models.resnet18(num_classes=3))
params(models.resnet34(num_classes=3))
params(models.resnet50(num_classes=3))
params(mvcnn())
params(mvcnn_att())


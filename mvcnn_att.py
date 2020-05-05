import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['MVCNN_ATT','mvcnn_att']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class MVCNN_ATT(nn.Module):
    def __init__(self, num_classes=1000):
        super(MVCNN_ATT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256*6*6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.V=nn.Sequential(
            nn.Linear(256*6*6,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1),
            nn.ReLU(inplace=True)
            )


    def forward(self, x):
        x = x.transpose(0, 1)
        
        view_pool = []
        weights=[]
        for v in x:
            v = self.features(v)
            v = v.view(v.size(0), 256 * 6 * 6) #(batch x 9216)
            w=self.V(v) #(batch x 1)
            view_pool.append(v)
            weights.append(w)
        weights=F.softmax(torch.cat(weights,1),1) #batch x 20
        #print(weights.shape)
        view_pool=torch.stack(view_pool,1) #batch x 20 x 9216 
        #print(view_pool.shape)

        #weights=weights.repeat(1,1,256*6*6) #batch x 20 x 9216
        weights = torch.stack([weights]*256*6*6,2)
        #print(weights.shape)
        pooled_view=(weights*view_pool).sum(1)

        pooled_view = self.classifier(pooled_view)
        return pooled_view


def mvcnn_att(pretrained=False, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MVCNN_ATT(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model
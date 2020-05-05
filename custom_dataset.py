from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import numpy as np
import torch

class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = ['0','1','2'],{'0':0,'1':1,'2':2}
        self.counts={'0':0,'1':0,'2':0}
        self.transform = transform
        self.target_transform = target_transform

        # root / <label>  / <train/test> / <item> / <view>.png
        # new structure ......root / <train/test> / <label> / <item_view>.png

        # for item in os.listdir(root):
        #     for label in os.listdir(root + '/'+ data_type + '/' + label ): # Label
            
        #         views = []
        #         for view in os.listdir(root + '/'  + data_type + '/' + label + view):
        #             views.append(root +  '/' + data_type + '/' + label + view)

        #         self.x.append(views)
        #         self.y.append(self.class_to_idx[label])
        # print("views",x)
        # print("labels",y)
        # for label in os.listdir(root): # Label
        #     for item in os.listdir(root + '/' + label + '/' + data_type):
        #         views = []
        #         for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
        #             views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)

        #         self.x.append(views)
        #         self.y.append(self.class_to_idx[label])
        # for datatype in ['val','train']:
        for label in os.listdir(root + '/'+ data_type):
            #print('label',label)
            #print('folder',root + '/'+ data_type +'/'+label)
            items= [ i.split('.')[0] for i in os.listdir(root + '/'+ data_type + '/' + label)]
            items=set(items)
            #print('#items',len(items))
            for item in items:
                views=[]
                # for j in range(20):
                #     views.append(root+'/'+data_type+'/'+label+'/'+item+'.ply_whiteshaded_v'+str(j)+'.png')
                for j in range(10):
                    views.append(root+'/'+data_type+'/'+label+'/'+item+'.ply_whiteshaded_v'+str(2*j+1)+'.png')
            
                self.x.append(views)
                self.y.append(int(label)) 
                self.counts[label]+=1   

        # print('customdataset size',len(self.x))
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view) #.convert('L')
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

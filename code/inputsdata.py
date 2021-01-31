from __future__ import division
import collections
import os.path as osp
import os
import errno
import numpy as np
import glob
import scipy.io as sio
import torch
import shutil
import random

import torch.utils.data
from torch.utils.data import Dataset, DataLoader
#from torch_geometric.data import DataLoader
from torchvision import transforms, utils
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
import torch.nn.functional as F
import torch_geometric.transforms as T
import os.path as osp
import torch

from MyData import Data


NUM = 8

label_dict = {}
with open('classInd.txt') as f:
    for line in f:
        (val, key) = line.split()
        label_dict[key.lower()] = int(val) - 1
        
        
        
class DataAug(object):
    def __init__(self, scales, p=0.5):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.p = p

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        
        if random.random() < self.p:
            pos = data.pos.clone()
            pos[:, 0] = 240 - pos[:, 0]
            data.pos = pos
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)
        
        
        
        
        
        
        
transform = T.Cartesian(cat=False)
class MyOwnDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        #self.transform = transform
        
    def __len__(self):
        return len(self.filename)
        
    def __getitem__(self, idx):
        
        raw_path = self.filename[idx]
        #print(raw_path)
        content = sio.loadmat(raw_path)
        
        graph_no = content['No'][0][0]
        interval = graph_no // NUM
        #print(interval)
        if interval < 2:
            selected_idx = [i for i in range(NUM)]
        else:
            selected_idx = [random.randint(i*interval+1, (i+1)*interval)-1 for i in range(NUM)]
        

        label_idx = torch.unsqueeze(torch.tensor([int(content['idx'])-1], dtype=torch.long),0)
        #print(label_idx)
        
        
        for i, idx in enumerate(selected_idx):
            graph = content['graph'][0, idx]
            feature = torch.tensor(graph['feature'])

            edge_index = torch.tensor(np.array(graph['edge'], np.int32), dtype=torch.long)
            pos = torch.tensor(graph['pos'])
            #print(pos.shape)
 
            data = Data(x=feature, edge_index=edge_index, pos=pos, y=label_idx.squeeze(0))
            
            data = transform(data)
                       
            if i == 0:
                g_x = data.x
                g_pos = data.pos
                g_edge_index = data.edge_index
                g_edge_attr = data.edge_attr 
                node_idx = np.ones(data.x.shape[0])*idx
                node_num = data.x.shape[0]
            else:
                g_x = torch.cat((g_x,data.x), 0)
                g_pos = torch.cat((g_pos,data.pos),0)
                g_edge_index = torch.cat((g_edge_index, data.edge_index),1)
                g_edge_attr = torch.cat((g_edge_attr, data.edge_attr),0)
                node_idx = np.append(node_idx, np.ones(data.x.shape[0])*idx)
                node_num = np.append(node_num, data.x.shape[0])
                
        node_idx = torch.tensor(node_idx, dtype=torch.int64)
        node_num = torch.tensor(node_num, dtype=torch.int64)
        g_data = Data(x=g_x, edge_index=g_edge_index, pos=g_pos, y=label_idx.squeeze(0), edge_attr=g_edge_attr, node_idx=node_idx, node_num=node_num)
       
        
        g_data = DataAug([0.8,0.999])(g_data)          # Data Augmentation: scale & left-right flip
  
        return g_data
                
                
if __name__ == '__main__':
    
    root = os.path.join('..', 'multi_graph')
    filename = glob.glob(os.path.join(root, '*.mat'))
    
    
    MNIST = MyOwnDataset(filename)
    #data = MNIST[0]
    #print(data)
    #for i in range(len(MNIST)):
        #sample = MNIST[i]
        #print(sample)
        
        
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = SplineConv(2, 64, dim=2, kernel_size=4)
            self.bn1 = torch.nn.BatchNorm1d(64)
            self.conv2 = SplineConv(64, 128, dim=2, kernel_size=4)
            self.bn2 = torch.nn.BatchNorm1d(128)
            self.conv3 = SplineConv(128, 256, dim=2, kernel_size=4)
            self.bn3 = torch.nn.BatchNorm1d(256)
            
        def forward(self, data):
            
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn1(data.x)
            cluster = voxel_grid(data.pos, data.batch, size=[2,2])
            data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn2(data.x)
            cluster = voxel_grid(data.pos, data.batch, size=[4,3])
            data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
            data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn3(data.x)
            cluster = voxel_grid(data.pos, data.batch, size=[8,6])
            x = max_pool_x(cluster, data.x, data.batch, size=900)
            
            x = x.reshape([8,30,30,256])
        
            return x
            
    
    dataloader = DataLoader(MNIST, batch_size=1, shuffle=True, collate_fn=lambda x:x)
    print(dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    for i, sample in enumerate(dataloader):
        #print(sample)
        
        for i_bt in range(len(sample)):
        
            data = sample[i_bt].to(device)
            #print(data.edge_attr,data.batch)
            end_point = model(data)
            
            #sio.savemat(i+'.mat', {'x': x})
            #print(i_bt, data.y)
            #print(data.index)
            #print(data.index.shape)
            #print(end_point)
            
        
        #print(i, sample_batch)
        #print(sample_batch[0].index)
            #print('======================')
    

                
                
                
                           